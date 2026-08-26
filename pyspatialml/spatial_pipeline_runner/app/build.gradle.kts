plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.bytedance.pico.pyspatialml.spatial_runner"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.bytedance.pico.pyspatialml.spatial_runner"
        minSdk = 34
        targetSdk = 34
        versionCode = 1
        versionName = "0.1.0"

        ndk { abiFilters.add("arm64-v8a") }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }

    lint {
        disable.add("ExpiredTargetSdkVersion")
    }
}

dependencies {
    implementation(platform("com.pico.spatial:bom:6.0.0"))
    implementation("com.pico.spatial.core:core")
    implementation("com.pico.spatial.tracking:tracking")
    implementation("com.pico.spatial.sense:sense")
    implementation("com.pico.spatial.ml:securemr:99.99.99-SNAPSHOT")
    implementation("com.pico.spatial.ml:readback:6.0.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
}
