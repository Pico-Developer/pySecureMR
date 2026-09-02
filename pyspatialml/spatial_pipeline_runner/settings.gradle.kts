pluginManagement {
    repositories {
        mavenLocal()
        maven(url = "https://maven.byted.org/repository/android_public")
        gradlePluginPortal()
        google()
        mavenCentral()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        mavenLocal()
        maven(url = "https://maven.byted.org/repository/android_public")
        google()
        mavenCentral()
    }
}

rootProject.name = "pySpatialML Spatial Pipeline Runner"
include(":app")
