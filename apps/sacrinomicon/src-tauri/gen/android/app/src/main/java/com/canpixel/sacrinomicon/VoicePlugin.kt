package com.canpixel.sacrinomicon

import android.app.Activity
import android.content.Intent
import androidx.activity.result.ActivityResult
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import app.tauri.annotation.Command
import app.tauri.annotation.ActivityCallback
import app.tauri.annotation.InvokeArg
import app.tauri.annotation.TauriPlugin
import app.tauri.plugin.Invoke
import app.tauri.plugin.JSObject
import app.tauri.plugin.Plugin
import java.util.Locale
import java.util.UUID
import java.io.File

@InvokeArg
class SpeechArgs { lateinit var text: String }
@InvokeArg
class ExportArgs { lateinit var path: String }

@TauriPlugin
class VoicePlugin(private val activity: Activity) : Plugin(activity) {
    private var engine: TextToSpeech? = null
    private var ready = false
    private var waiting: (() -> Unit)? = null
    private var current: Invoke? = null
    private var utteranceId: String? = null

    @Command
    fun saveExport(invoke: Invoke) {
        try {
            val args = invoke.parseArgs(ExportArgs::class.java)
            val file = File(args.path).canonicalFile
            val exports = File(activity.cacheDir, "exports").canonicalFile
            require(file.parentFile == exports && file.isFile) { "Export is unavailable" }
            val intent = Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
                addCategory(Intent.CATEGORY_OPENABLE)
                type = "application/zip"
                putExtra(Intent.EXTRA_TITLE, file.name)
            }
            startActivityForResult(invoke, intent, "saveExportResult")
        } catch (error: Exception) { invoke.reject(error.message ?: "Could not open export destination") }
    }

    @ActivityCallback
    fun saveExportResult(invoke: Invoke, result: ActivityResult) {
        if (result.resultCode == Activity.RESULT_CANCELED) { invoke.resolve(JSObject().put("saved", false)); return }
        val uri = result.data?.data
        if (result.resultCode != Activity.RESULT_OK || uri == null) { invoke.reject("No export destination was selected"); return }
        Thread {
            try {
                val file = File(invoke.parseArgs(ExportArgs::class.java).path).canonicalFile
                require(file.parentFile == File(activity.cacheDir, "exports").canonicalFile && file.isFile) { "Export is unavailable" }
                activity.contentResolver.openOutputStream(uri, "wt").use { output ->
                    requireNotNull(output) { "Could not write to this destination" }
                    file.inputStream().use { input -> input.copyTo(output, 65536) }
                }
                invoke.resolve(JSObject().put("saved", true))
            } catch (error: Exception) { invoke.reject(error.message ?: "Could not save library export") }
        }.start()
    }

    @Command
    fun speak(invoke: Invoke) {
        val args = invoke.parseArgs(SpeechArgs::class.java)
        activity.runOnUiThread {
            current?.resolve(JSObject())
            current = invoke
            val action = action@{
                val tts = engine ?: return@action
                val voice = tts.voices?.firstOrNull { !it.isNetworkConnectionRequired && it.locale.language == Locale.getDefault().language }
                    ?: tts.voices?.firstOrNull { !it.isNetworkConnectionRequired && it.locale.language == "en" }
                if (voice == null) { invoke.reject("Install an offline system voice to use read-aloud."); current = null; return@action }
                tts.voice = voice
                tts.setSpeechRate(0.95f)
                val id = UUID.randomUUID().toString(); utteranceId = id
                tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                    override fun onStart(id: String?) {}
                    override fun onDone(id: String?) { if (id == utteranceId) { current?.resolve(JSObject()); current = null } }
                    @Deprecated("Android callback")
                    override fun onError(id: String?) { if (id == utteranceId) { current?.reject("Offline speech failed"); current = null } }
                })
                val chunks = args.text.chunked(3000)
                for ((index, chunk) in chunks.withIndex()) {
                    val result = tts.speak(chunk, if (index == 0) TextToSpeech.QUEUE_FLUSH else TextToSpeech.QUEUE_ADD, null, if (index == chunks.lastIndex) id else "$id-$index")
                    if (result == TextToSpeech.ERROR) { invoke.reject("Offline speech failed"); current = null; break }
                }
            }
            if (ready) action() else {
                waiting = action
                if (engine == null) engine = TextToSpeech(activity) { status ->
                    ready = status == TextToSpeech.SUCCESS
                    if (ready) { waiting?.invoke(); waiting = null }
                    else { current?.reject("The system speech engine is unavailable"); current = null; waiting = null }
                }
            }
        }
    }
    @Command
    fun stop(invoke: Invoke) {
        activity.runOnUiThread {
            waiting = null; engine?.stop(); current?.resolve(JSObject()); current = null; utteranceId = null
            invoke.resolve(JSObject())
        }
    }
}
