package com.bytedance.pico.pyspatialml.spatialrunner

import android.app.Activity
import android.content.Context
import android.content.ContextWrapper
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.SharedMemory
import android.os.SystemClock
import android.util.Log
import android.widget.TextView
import com.pico.spatial.ml.readback.readbackContentSuspend
import com.pico.spatial.ml.securemr.GlobalTensor
import com.pico.spatial.ml.securemr.Pipeline
import com.pico.spatial.ml.securemr.PipelinePackageBundle
import com.pico.spatial.ml.securemr.SpatialMLInstance
import com.pico.spatial.ml.securemr.SpatialMLSession
import com.pico.spatial.ml.securemr.loadPipelinePackageFromAssets
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

class MainActivity : Activity() {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(TextView(this).apply { text = "pySpatialML Spatial Runner" })
        scope.launch {
            try {
                SpatialRunner(applicationContext).run()
            } catch (cancellation: CancellationException) {
                throw cancellation
            } catch (error: Throwable) {
                Log.e(TAG, "runner_error ${error.message}", error)
                writeStatus(
                    File(filesDir, "outputs"),
                    JSONObject()
                        .put("state", "error")
                        .put("mode", "spatial")
                        .put("error", error.message ?: error.javaClass.name),
                )
            }
        }
    }

    override fun onDestroy() {
        scope.cancel()
        super.onDestroy()
    }

    private companion object {
        private const val TAG = "pySpatialML Spatial runner"
    }
}

private class SpatialRunner(private val context: Context) {
    private val packageDir = propPath("package", File(context.filesDir, "package"))
    private val packageZip = propPath("package_zip", File(context.filesDir, "package.zip"))
    private val outputDir = propPath("output", File(context.filesDir, "outputs"))
    private val inputPath = prop("input", "")
    private val assetRoot = prop("asset_root", "package").trim('/')
    private val useVst = propBool("use_vst", inputPath.isBlank())
    private val loop = propBool("loop", false)
    private val dumpAll = propBool("dump_all", false)
    private val intervalMs = propInt("interval_ms", 50).coerceAtLeast(1)
    private val requestedPipelines = csv(prop("pipelines", ""))
    private val inputMemories = mutableListOf<SharedMemory>()

    suspend fun run() {
        withContext(Dispatchers.IO) {
            outputDir.deleteRecursively()
            outputDir.mkdirs()
        }
        require(packageDir.isDirectory) { "Package directory not found: ${packageDir.absolutePath}" }
        require(packageZip.isFile) { "Package zip not found: ${packageZip.absolutePath}" }

        val manifest = readJson(File(packageDir, "manifest.json"))
        val instance = SpatialMLInstance.create(ZipAssetContext(context, packageZip))
        val readyDeadline = SystemClock.elapsedRealtime() + READY_TIMEOUT_MS
        while (!instance.ready && SystemClock.elapsedRealtime() < readyDeadline) {
            delay(READY_POLL_MS)
        }
        check(instance.ready) { "SpatialML instance was not ready after ${READY_TIMEOUT_MS}ms" }

        val session =
            instance.createSession(
                SpatialMLSession.InitInfo(
                        imageWidth = propInt("camera_width", DEFAULT_CAMERA_WIDTH),
                        imageHeight = propInt("camera_height", DEFAULT_CAMERA_HEIGHT),
                        containerWidth = propInt("container_width", DEFAULT_CONTAINER_WIDTH),
                        containerHeight = propInt("container_height", DEFAULT_CONTAINER_HEIGHT),
                        containerDepth = propInt("container_depth", DEFAULT_CONTAINER_DEPTH),
                        containerType = SpatialMLSession.ContainerType.VOLUMETRIC,
                    )
                    .addPortal()
            ) ?: error("SpatialML session creation returned null")

        val bundle = session.loadPipelinePackageFromAssets(assetRoot)
        val tensorSpecs = readTensorSpecs(manifest)
        val pipelineOrder = pipelineOrder(manifest, bundle)
        bindInputs(bundle, tensorSpecs)
        Log.i(
            TAG,
            "loaded package=${packageDir.absolutePath} pipelines=${pipelineOrder.joinToString()}",
        )

        var iteration = 0
        do {
            val startedAt = SystemClock.elapsedRealtime()
            var waitFor: Pipeline.RunTask? = null
            for (pipelineId in pipelineOrder) {
                val pipeline = bundle.pipelines[pipelineId] ?: error("Unknown pipeline '$pipelineId'")
                waitFor = pipeline.pipeline.submit(pipeline.submitBindings, null, waitFor)
            }
            iteration += 1
            writeStatus(
                outputDir,
                JSONObject()
                    .put("state", if (loop) "running" else "submitted")
                    .put("mode", "spatial")
                    .put("iteration", iteration)
                    .put("loop", loop)
                    .put("use_vst", useVst)
                    .put("runtime_modes", manifest.optJSONObject("runtime")?.optJSONArray("supported_modes") ?: JSONArray())
                    .put("pipelines", JSONArray(pipelineOrder)),
            )
            Log.i(TAG, "iteration $iteration submitted")
            delay(READBACK_SETTLE_MS)
            if (!readOutputs(bundle, tensorSpecs, pipelineOrder, iteration, startedAt)) {
                return
            }
            if (loop) {
                delay(intervalMs.toLong())
            }
        } while (loop && currentCoroutineContext().isActive)
    }

    private fun pipelineOrder(manifest: JSONObject, bundle: PipelinePackageBundle): List<String> {
        if (requestedPipelines.isNotEmpty()) return requestedPipelines
        val manifestPipelines = manifest.optJSONArray("pipelines") ?: JSONArray()
        val ordered = mutableListOf<String>()
        for (index in 0 until manifestPipelines.length()) {
            val id = manifestPipelines.optJSONObject(index)?.optString("id").orEmpty()
            if (id.isNotBlank()) ordered += id
        }
        return ordered.ifEmpty { bundle.pipelines.keys.toList() }
    }

    private fun bindInputs(bundle: PipelinePackageBundle, tensorSpecs: Map<String, TensorSpec>) {
        if (useVst) {
            Log.i(TAG, "using device VST inputs")
            return
        }
        val bindableInputNames =
            bundle.pipelines.values
                .flatMap { it.inputs }
                .filter { name -> tensorSpecs[name]?.isAssetReference() != true }
                .toSet()
        if (bindableInputNames.isEmpty()) {
            Log.i(TAG, "no host-provided tensor inputs to bind")
            return
        }
        require(inputPath.isNotBlank()) { "image/raw input mode requires input path" }
        val inputRoot = File(inputPath)
        val bareImageInput = !inputRoot.isDirectory && inputRoot.isImageFile()
        val cache = linkedMapOf<String, ByteArray>()
        for ((pipelineId, pipeline) in bundle.pipelines) {
            for (inputName in pipeline.inputs) {
                val tensor = bundle.globalTensors[inputName] ?: continue
                val spec = tensorSpecs[inputName] ?: continue
                val file = resolveInputPath(inputRoot, inputName)
                val bytes =
                    when {
                        file != null && file.isImageFile() && spec.isImageTensor() ->
                            cache.getOrPut("${file.absolutePath}|$spec") { file.decodeImage(spec) }
                        file != null && !bareImageInput -> file.readBytes()
                        inputRoot.isDirectory && spec.isImageTensor() ->
                            resolveDefaultInput(inputRoot)?.let { image ->
                                cache.getOrPut("${image.absolutePath}|$spec") { image.decodeImage(spec) }
                            } ?: continue
                        !inputRoot.isDirectory && spec.isImageTensor() ->
                            cache.getOrPut("${inputRoot.absolutePath}|$spec") { inputRoot.decodeImage(spec) }
                        spec.isDefaultable(inputName) -> spec.defaultBytes(inputName)
                        else -> continue
                    }
                val expected = spec.byteCount()
                require(expected == 0 || bytes.size == expected) {
                    "input $inputName has ${bytes.size} bytes but tensor expects $expected bytes"
                }
                tensor.setBytes(bytes, "input_${pipelineId}_$inputName")
                Log.i(TAG, "bound input $inputName for pipeline $pipelineId bytes=${bytes.size}")
            }
        }
    }

    private suspend fun readOutputs(
        bundle: PipelinePackageBundle,
        tensorSpecs: Map<String, TensorSpec>,
        pipelineOrder: List<String>,
        iteration: Int,
        startedAt: Long,
    ): Boolean {
        val metadata = JSONArray()
        val failedOutputs = JSONArray()
        var outputsExpected = 0
        var outputsWritten = 0
        for (pipelineId in pipelineOrder) {
            val pipeline = bundle.pipelines[pipelineId] ?: continue
            val targetNames = linkedSetOf<String>()
            for (outputName in pipeline.outputs) {
                targetNames += outputName
            }
            if (dumpAll) {
                for (inputName in pipeline.inputs) {
                    targetNames += inputName
                }
                bundle.detectionTensor?.let { targetNames += it }
            }
            for (tensorName in targetNames) {
                val isOutput = pipeline.outputs.contains(tensorName)
                val tensor = bundle.globalTensors[tensorName]
                val spec = tensorSpecs[tensorName]
                if (isOutput && spec?.isAssetReference() != true) outputsExpected += 1
                if (tensor == null || spec == null) {
                    if (isOutput && spec?.isAssetReference() != true) {
                        failedOutputs.put(
                            JSONObject()
                                .put("pipeline", pipelineId)
                                .put("tensor", tensorName)
                                .put("error", "required output tensor was not materialized")
                        )
                    }
                    continue
                }
                if (spec.isAssetReference()) continue
                val bytes = try {
                    readTensorBytes(tensor)
                } catch (error: CancellationException) {
                    throw error
                } catch (error: Throwable) {
                    if (isOutput) {
                        failedOutputs.put(
                            JSONObject()
                                .put("pipeline", pipelineId)
                                .put("tensor", tensorName)
                                .put("error", error.message ?: error.javaClass.name)
                        )
                    } else {
                        Log.w(TAG, "optional readback skipped pipeline=$pipelineId tensor=$tensorName", error)
                    }
                    continue
                }
                val filename = "${safeName(pipelineId)}_${safeName(tensorName)}_$iteration.bin"
                withContext(Dispatchers.IO) {
                    File(outputDir, filename).writeBytes(bytes)
                }
                metadata.put(
                    JSONObject()
                        .put("file", filename)
                        .put("pipeline", pipelineId)
                        .put("tensor", tensorName)
                        .put("dtype", spec.dtypeName())
                        .put("data_type", spec.dataType)
                        .put("channels", spec.channels)
                        .put("bytes", bytes.size)
                        .put("is_output", isOutput)
                        .put("shape", JSONArray(spec.dimensions.toList()))
                )
                if (isOutput) outputsWritten += 1
                Log.i(TAG, "wrote readback $filename bytes=${bytes.size}")
            }
        }
        val complete = failedOutputs.length() == 0
        writeStatus(
            outputDir,
            JSONObject()
                .put("state", if (complete) "complete" else "error")
                .put("mode", "spatial")
                .put("iteration", iteration)
                .put("loop", loop)
                .put("use_vst", useVst)
                .put("total_elapsed_ms", SystemClock.elapsedRealtime() - startedAt)
                .put("runtime_modes", JSONArray(listOf("spatial")))
                .put("pipelines", JSONArray(pipelineOrder))
                .put("outputs_expected", outputsExpected)
                .put("outputs_written", outputsWritten)
                .put("failed_outputs", failedOutputs)
                .put("outputs_metadata", metadata)
        )
        if (!complete) {
            Log.e(TAG, "required output readback failed expected=$outputsExpected written=$outputsWritten failures=$failedOutputs")
        }
        return complete
    }

    private suspend fun readTensorBytes(tensor: GlobalTensor): ByteArray =
        try {
            tensor.readbackContentSuspend().use { content ->
                val duplicate = content.buffer.duplicate()
                duplicate.rewind()
                val bytes = ByteArray(duplicate.remaining())
                duplicate.get(bytes)
                bytes
            }
        } catch (error: CancellationException) {
            throw error
        } catch (error: Throwable) {
            Log.w(TAG, "readback failed ${error.message}", error)
            throw error
        }

    private fun readTensorSpecs(manifest: JSONObject): Map<String, TensorSpec> {
        val result = linkedMapOf<String, TensorSpec>()
        val pipelines = manifest.getJSONArray("pipelines")
        for (index in 0 until pipelines.length()) {
            val path = pipelines.getJSONObject(index).getString("path")
            val pipelineJson = readJson(File(packageDir, path))
            val tensors = pipelineJson.getJSONObject("tensors")
            val names = tensors.keys()
            while (names.hasNext()) {
                val name = names.next()
                result.putIfAbsent(name, TensorSpec.fromJson(tensors.getJSONObject(name)))
            }
        }
        return result
    }

    private fun resolveInputPath(root: File, tensorName: String): File? {
        if (root.isDirectory) {
            for (ext in INPUT_EXTENSIONS) {
                val candidate = File(root, "${safeName(tensorName)}$ext")
                if (candidate.isFile) return candidate
            }
            return null
        }
        return root.takeIf { it.isFile }
    }

    private fun resolveDefaultInput(root: File): File? =
        INPUT_EXTENSIONS.map { File(root, "__default$it") }.firstOrNull { it.isFile }

    private fun File.isImageFile(): Boolean =
        extension.lowercase(Locale.US) in setOf("jpg", "jpeg", "png")

    private fun File.decodeImage(spec: TensorSpec): ByteArray {
        val source = BitmapFactory.decodeFile(absolutePath)
            ?: error("failed to decode image input: $absolutePath")
        val width = spec.dimensions.getOrNull(0) ?: source.width
        val height = spec.dimensions.getOrNull(1) ?: source.height
        val cropped = source.centerCrop(width, height)
        val scaled =
            if (cropped.width == width && cropped.height == height) cropped
            else Bitmap.createScaledBitmap(cropped, width, height, true)
        val bytes = ByteArray(width * height * spec.channels)
        val row = IntArray(width)
        var offset = 0
        for (y in 0 until height) {
            scaled.getPixels(row, 0, width, 0, y, width, 1)
            for (pixel in row) {
                val r = (pixel shr 16).toByte()
                val g = (pixel shr 8).toByte()
                val b = pixel.toByte()
                val a = (pixel ushr 24).toByte()
                when (spec.channels) {
                    1 -> bytes[offset++] = r
                    3 -> {
                        bytes[offset++] = r
                        bytes[offset++] = g
                        bytes[offset++] = b
                    }
                    4 -> {
                        bytes[offset++] = r
                        bytes[offset++] = g
                        bytes[offset++] = b
                        bytes[offset++] = a
                    }
                }
            }
        }
        if (scaled !== cropped) scaled.recycle()
        if (cropped !== source) cropped.recycle()
        source.recycle()
        return bytes
    }

    private fun Bitmap.centerCrop(targetWidth: Int, targetHeight: Int): Bitmap {
        if (targetWidth <= 0 || targetHeight <= 0) return this
        val sourceAspect = width.toFloat() / height.toFloat()
        val targetAspect = targetWidth.toFloat() / targetHeight.toFloat()
        val cropWidth: Int
        val cropHeight: Int
        if (sourceAspect > targetAspect) {
            cropHeight = height
            cropWidth = (height * targetAspect).toInt().coerceIn(1, width)
        } else {
            cropWidth = width
            cropHeight = (width / targetAspect).toInt().coerceIn(1, height)
        }
        if (cropWidth == width && cropHeight == height) return this
        val left = ((width - cropWidth) / 2).coerceAtLeast(0)
        val top = ((height - cropHeight) / 2).coerceAtLeast(0)
        return Bitmap.createBitmap(this, left, top, cropWidth, cropHeight)
    }

    private fun GlobalTensor.setBytes(bytes: ByteArray, name: String) {
        val memory = SharedMemory.create(name, bytes.size)
        val buffer = memory.mapReadWrite()
        try {
            buffer.put(bytes)
            buffer.rewind()
        } finally {
            SharedMemory.unmap(buffer)
        }
        tensorResource = memory
        inputMemories += memory
    }

    private companion object {
        private const val TAG = "pySpatialML Spatial runner"
        private const val PROP_PREFIX = "debug.pyspatialml.spatial_runner"
        private const val EMPTY_PROP = "__pyspatialml_empty__"
        private const val DEFAULT_CAMERA_WIDTH = 580
        private const val DEFAULT_CAMERA_HEIGHT = 326
        private const val DEFAULT_CONTAINER_WIDTH = 1000
        private const val DEFAULT_CONTAINER_HEIGHT = 1000
        private const val DEFAULT_CONTAINER_DEPTH = 10
        private const val READY_TIMEOUT_MS = 10_000L
        private const val READY_POLL_MS = 100L
        private const val READBACK_SETTLE_MS = 40L
        private val INPUT_EXTENSIONS = listOf(".bin", ".raw", ".jpg", ".jpeg", ".png")

        fun writeStatus(outputDir: File, status: JSONObject) {
            outputDir.mkdirs()
            File(outputDir, "status.json").writeText(status.toString(2))
        }

        private fun prop(key: String, defaultValue: String): String {
            val value =
                try {
                    Class.forName("android.os.SystemProperties")
                        .getMethod("get", String::class.java, String::class.java)
                        .invoke(null, "$PROP_PREFIX.$key", defaultValue) as String
                } catch (_: ReflectiveOperationException) {
                    defaultValue
                }
            return if (value == EMPTY_PROP) "" else value
        }

        private fun propPath(key: String, defaultValue: File): File =
            prop(key, defaultValue.absolutePath).takeIf { it.isNotBlank() }?.let(::File) ?: defaultValue

        private fun propBool(key: String, defaultValue: Boolean): Boolean =
            when (prop(key, defaultValue.toString()).lowercase(Locale.US)) {
                "1", "true", "y", "yes", "on" -> true
                "0", "false", "n", "no", "off" -> false
                else -> defaultValue
            }

        private fun propInt(key: String, defaultValue: Int): Int =
            prop(key, defaultValue.toString()).toIntOrNull() ?: defaultValue

        private fun csv(value: String): List<String> =
            value.split(',').map { it.trim() }.filter { it.isNotEmpty() }

        private fun readJson(file: File): JSONObject = JSONObject(file.readText())

        private fun safeName(value: String): String = value.replace(Regex("[^A-Za-z0-9_.-]"), "_")
    }

}

private fun writeStatus(outputDir: File, status: JSONObject) {
    outputDir.mkdirs()
    File(outputDir, "status.json").writeText(status.toString(2))
}

private data class TensorSpec(
    val dimensions: IntArray,
    val channels: Int,
    val dataType: Int,
    val tensorType: String,
    val asset: String?,
    val isGltf: Boolean,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as TensorSpec
        return channels == other.channels &&
            dataType == other.dataType &&
            dimensions.contentEquals(other.dimensions)
    }

    override fun hashCode(): Int {
        var result = dimensions.contentHashCode()
        result = 31 * result + channels
        result = 31 * result + dataType
        return result
    }

    fun byteCount(): Int = byteSize() * dimensions.fold(1) { acc, value -> acc * value } * channels

    fun dtypeName(): String =
        when (dataType) {
            1 -> "uint8"
            2 -> "int8"
            3 -> "uint16"
            4 -> "int16"
            5 -> "int32"
            6 -> "float32"
            7 -> "float64"
            else -> "unknown"
        }

    fun isImageTensor(): Boolean =
        dataType == 1 && dimensions.size >= 2 && channels in setOf(1, 3, 4)

    fun isAssetReference(): Boolean =
        isGltf || tensorType == "gltf" || !asset.isNullOrBlank()

    fun isDefaultable(name: String): Boolean =
        (name.contains("timestamp") && dataType == 5) ||
            (name.contains("camera_matrix") &&
                dataType == 6 &&
                channels == 1 &&
                dimensions.contentEquals(intArrayOf(3, 3)))

    fun defaultBytes(name: String): ByteArray {
        val bytes = ByteArray(byteCount())
        if (name.contains("camera_matrix") && dataType == 6 && bytes.size >= 9 * Float.SIZE_BYTES) {
            ByteBuffer.wrap(bytes).order(ByteOrder.nativeOrder()).apply {
                putFloat(0, 1.0f)
                putFloat(4 * Float.SIZE_BYTES, 1.0f)
                putFloat(8 * Float.SIZE_BYTES, 1.0f)
            }
        }
        return bytes
    }

    private fun byteSize(): Int =
        when (dataType) {
            1, 2 -> 1
            3, 4 -> 2
            5, 6 -> 4
            7 -> 8
            else -> 0
        }

    companion object {
        fun fromJson(json: JSONObject): TensorSpec {
            val dimsJson = json.optJSONArray("dimensions") ?: JSONArray()
            val dims = IntArray(dimsJson.length()) { index -> dimsJson.getInt(index) }
            return TensorSpec(
                dimensions = dims,
                channels = json.optInt("channels", 1),
                dataType = parseDataType(json.opt("data_type")),
                tensorType = json.optString("tensor_type", json.optString("type", "")).lowercase(Locale.US),
                asset = json.optString("asset", "").takeIf { it.isNotBlank() },
                isGltf = json.optBoolean("is_gltf", false),
            )
        }

        private fun parseDataType(value: Any?): Int =
            when (value) {
                is Number -> value.toInt()
                is String ->
                    when (value.lowercase(Locale.US)) {
                        "uint8", "u8" -> 1
                        "int8", "i8" -> 2
                        "uint16", "u16" -> 3
                        "int16", "i16" -> 4
                        "int32", "i32" -> 5
                        "float32", "fp32", "f32" -> 6
                        "float64", "fp64", "f64" -> 7
                        else -> 6
                    }
                else -> 6
            }
    }
}

private class ZipAssetContext(base: Context, packageZip: File) : ContextWrapper(base) {
    private val zipAssets: AssetManager = AssetManager::class.java.getDeclaredConstructor().newInstance()

    init {
        val cookie =
            AssetManager::class.java
                .getMethod("addAssetPath", String::class.java)
                .invoke(zipAssets, packageZip.absolutePath) as Int
        require(cookie != 0) { "Failed to add package zip as asset path: ${packageZip.absolutePath}" }
    }

    override fun getAssets(): AssetManager = zipAssets
}
