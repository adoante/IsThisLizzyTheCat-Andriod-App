package com.example.isthislizzythecat

import android.Manifest
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.example.isthislizzythecat.ml.IsLizzyTheCat
import com.example.isthislizzythecat.ui.theme.IsThisLizzyTheCatTheme
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors
import kotlin.math.exp

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val requestPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                setContent {
                    IsThisLizzyTheCatTheme {
                        Surface(modifier = Modifier.fillMaxSize()) {
                            RealTimeCameraScreen()
                        }
                    }
                }
            } else {
                Log.e("Permission", "Camera permission not granted")
            }
        }

        requestPermission.launch(Manifest.permission.CAMERA)
    }
}

@Composable
fun RealTimeCameraScreen() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    var resultText by remember { mutableStateOf("Analyzing...") }

    AndroidView(
        factory = { ctx ->
            val previewView = PreviewView(ctx)
            val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)

            cameraProviderFuture.addListener({
                val cameraProvider = cameraProviderFuture.get()

                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

                val analyzer = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also {
                        it.setAnalyzer(cameraExecutor) { imageProxy ->
                            Log.d("CameraX", "Processing frame...")
                            val bitmap = imageProxy.okBitmap()
                            if (bitmap != null) {
                                resultText = runModelOnImage(context, bitmap)
                            }
                            imageProxy.close()
                        }
                    }

                val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    cameraSelector,
                    preview,
                    analyzer
                )
            }, ContextCompat.getMainExecutor(ctx))

            previewView
        },
        modifier = Modifier.fillMaxSize()
    )

    Box(
        contentAlignment = Alignment.BottomCenter,
        modifier = Modifier.fillMaxSize().padding(24.dp)
    ) {
        Box(
            modifier = Modifier
                .background(Color.Black.copy(alpha = 0.7f), shape = RoundedCornerShape(12.dp))
                .padding(horizontal = 16.dp, vertical = 8.dp)
        ) {
            Text(
                text = resultText,
                color = Color.White,
                style = MaterialTheme.typography.headlineSmall
            )
        }
    }
}

fun ImageProxy.okBitmap(): Bitmap? {
    val yBuffer = planes[0].buffer
    val uBuffer = planes[1].buffer
    val vBuffer = planes[2].buffer

    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()

    val nv21 = ByteArray(ySize + uSize + vSize)
    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)

    val yuvImage = YuvImage(
        nv21,
        ImageFormat.NV21,
        width, height,
        null
    )

    val out = ByteArrayOutputStream()
    yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, out)
    val imageBytes = out.toByteArray()
    return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
}

fun runModelOnImage(context: Context, bitmap: Bitmap): String {
    val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
    val byteBuffer = ByteBuffer.allocateDirect(4 * 3 * 224 * 224)
    byteBuffer.order(ByteOrder.nativeOrder())

    val intValues = IntArray(224 * 224)
    resized.getPixels(intValues, 0, 224, 0, 0, 224, 224)

    val r = FloatArray(224 * 224)
    val g = FloatArray(224 * 224)
    val b = FloatArray(224 * 224)

    for (i in intValues.indices) {
        val pixel = intValues[i]
        r[i] = ((pixel shr 16 and 0xFF) / 255.0f)
        g[i] = ((pixel shr 8 and 0xFF) / 255.0f)
        b[i] = ((pixel and 0xFF) / 255.0f)
    }

    for (channel in listOf(r, g, b)) {
        for (value in channel) {
            byteBuffer.putFloat(value)
        }
    }

    val model = IsLizzyTheCat.newInstance(context)
    return try {
        val inputFeature = TensorBuffer.createFixedSize(intArrayOf(1, 3, 224, 224), DataType.FLOAT32)
        inputFeature.loadBuffer(byteBuffer)

        val output = model.process(inputFeature).outputFeature0AsTensorBuffer
        val probs = softmax(output.floatArray)
        val predictedIndex = probs.indices.maxByOrNull { probs[it] } ?: -1

        when {
            probs[0] > 0.9f -> "It's Lizzy! Confidence: ${(probs[0] * 100).toInt()}%"
            probs[1] > 0.9f -> "Not Lizzy. Confidence: ${(probs[1] * 100).toInt()}%"
            else -> "Uncertain result. (Lizzy: ${(probs[0] * 100).toInt()}%, Not Lizzy: ${(probs[1] * 100).toInt()}%)"
        }
    } finally {
        model.close()
    }
}

fun softmax(logits: FloatArray): FloatArray {
    val maxLogit = logits.maxOrNull() ?: 0f
    val exps = logits.map { exp((it - maxLogit).toDouble()) }
    val sumExp = exps.sum()
    return exps.map { (it / sumExp).toFloat() }.toFloatArray()
}