package com.canpixel.sacrinomicon

import android.view.View
import android.view.ViewGroup
import android.webkit.WebView
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.json.JSONObject
import org.json.JSONArray
import java.io.ByteArrayOutputStream
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/** Actual WebView -> Tauri -> SQLite commands. No mocked IPC or permissions. */
@RunWith(AndroidJUnit4::class)
class NativeArchiveTest {
    private fun web(view: View): WebView? {
        if (view is WebView) return view
        if (view is ViewGroup) for (i in 0 until view.childCount) web(view.getChildAt(i))?.let { return it }
        return null
    }
    private fun evaluate(view: WebView, script: String): String {
        val latch = CountDownLatch(1); var result = "null"
        InstrumentationRegistry.getInstrumentation().runOnMainSync {
            view.evaluateJavascript(script) { result = it; latch.countDown() }
        }
        assertTrue("JavaScript callback timed out", latch.await(10, TimeUnit.SECONDS))
        return result
    }
    @Test fun nativeImportCorrectionsSearchAndVoiceCleanup() {
        // Tauri owns a process-wide native event loop. ActivityScenario.close()
        // destroys that loop while instrumentation keeps the process alive.
        // Keep the activity alive; the test runner terminates its process instead.
        val scenario = ActivityScenario.launch(MainActivity::class.java)
        run {
            var view: WebView? = null
            scenario.onActivity { view = web(it.window.decorView) }
            val browser = requireNotNull(view)
            val deadline = System.currentTimeMillis() + 30000
            while (evaluate(browser, "typeof window.__TAURI_INTERNALS__?.invoke === 'function' && document.body.innerText.includes('Oracle')") != "true") {
                assertTrue("Native frontend did not become ready", System.currentTimeMillis() < deadline)
                Thread.sleep(100)
            }
            val manifest = """{"schemaVersion":1,"revision":"revision:native-smoke","exportedAt":1,"source":{"id":"source:native-smoke","title":"Native verification fixture","uri":"","publisher":"Test fixture","durationSeconds":60,"capturedAt":1,"summary":null},"embeddingModel":"bge-small-en-v1.5-q8_0","evidence":[{"id":"evidence:native-smoke","sourceId":"source:native-smoke","start":1,"end":10,"text":"The source calls the account Ball. This is an attributed assertion.","kind":"speech","provenance":{"method":"test_fixture","model":null,"generatedAt":1,"originalAsset":null},"vector":[]}],"claims":[],"concepts":[],"relationships":[],"assets":[]}"""
            val buffer = ByteArrayOutputStream()
            ZipOutputStream(buffer).use { zip -> zip.putNextEntry(ZipEntry("manifest.json")); zip.write(manifest.toByteArray()); zip.closeEntry() }
            val bytes = JSONArray(buffer.toByteArray().map { it.toInt() and 255 }).toString()
            evaluate(browser, """
                window.__sacriNativeTest = {done:false};
                (async () => {
                  const call = window.__TAURI_INTERNALS__.invoke;
                  const id = await call('begin_import');
                  await call('append_import', {id,offset:0,bytes:$bytes});
                  const report = await call('finish_import',{id});
                  const e = 'evidence:native-smoke';
                  await call('correct_reading',{evidenceId:e,corrected:'The source calls the account Baal. This is an attributed assertion.'});
                  const original = await call('evidence_detail',{id:e});
                  const reading = await call('evidence_reading',{id:e});
                  const hits = await call('search',{query:'Baal',sourceId:'source:native-smoke'});
                  const cancelled = await call('begin_import');
                  await call('cancel_import',{id:cancelled});
                  let cancelledRejected = false;
                  try { await call('finish_import',{id:cancelled}); } catch { cancelledRejected = true; }
                  await call('voice_mode',{enabled:false});
                  await call('stop_speech');
                  const models = await call('model_status');
                  window.__sacriNativeTest={done:true,report,original:original.text,reading,hits:hits.length,cancelledRejected,models:models.length};
                })().catch(error=>{window.__sacriNativeTest={done:true,error:String(error)}});
            """.trimIndent())
            val finish = System.currentTimeMillis() + 60000
            while (evaluate(browser, "window.__sacriNativeTest.done") != "true") {
                assertTrue("Native commands timed out", System.currentTimeMillis() < finish)
                Thread.sleep(100)
            }
            val result = JSONObject(evaluate(browser, "window.__sacriNativeTest"))
            assertFalse(result.toString(), result.has("error"))
            assertEquals("source:native-smoke", result.getJSONObject("report").getJSONArray("sourceIds").getString(0))
            assertTrue(result.getString("original").contains("Ball"))
            assertTrue(result.getString("reading").contains("Baal"))
            assertTrue(result.getInt("hits") > 0)
            assertTrue(result.getBoolean("cancelledRejected"))
            assertEquals(3, result.getInt("models"))
        }
    }
}
