import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

void main() {
  runApp(const CypriotOcrApp());
}

class CypriotOcrApp extends StatelessWidget {
  const CypriotOcrApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Cypriot Syllabary OCR',
      theme: ThemeData(primarySwatch: Colors.amber, useMaterial3: true),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  Interpreter? _interpreter;
  File? _selectedImage;
  String _extractedText = "";
  bool _isLoading = false;

  // 55 Cypriot Syllabary character labels matching dataset yaml
  final List<String> classNames = [
    'a',
    'e',
    'i',
    'ja',
    'jo',
    'ka',
    'ke',
    'ki',
    'ko',
    'ku',
    'la',
    'le',
    'li',
    'lo',
    'lu',
    'ma',
    'me',
    'mi',
    'mo',
    'mu',
    'na',
    'ne',
    'ni',
    'no',
    'nu',
    'o',
    'pa',
    'pe',
    'pi',
    'po',
    'pu',
    'ra',
    're',
    'ri',
    'ro',
    'ru',
    'sa',
    'se',
    'si',
    'so',
    'su',
    'ta',
    'te',
    'ti',
    'to',
    'tu',
    'u',
    'wa',
    'we',
    'wi',
    'wo',
    'xa',
    'xe',
    'za',
    'zo',
  ];

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset(
        'assets/cypriot_yolo_finetuned_best_float32.tflite',
      );
      debugPrint("YOLO TFLite loaded successfully.");
    } catch (e) {
      debugPrint("Failed to load model: $e");
    }
  }

  Future<void> _pickAndProcessImage() async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);

    if (pickedFile == null || _interpreter == null) return;

    setState(() {
      _selectedImage = File(pickedFile.path);
      _isLoading = true;
      _extractedText = "";
    });

    // Run processing on background image tensor
    String resultWord = await _runInference(_selectedImage!);

    setState(() {
      _extractedText = resultWord.isNotEmpty ? resultWord : "No text detected.";
      _isLoading = false;
    });
  }

  Future<String> _runInference(File imageFile) async {
    // 1. Read and resize image to 640x640 required by YOLO
    Uint8List bytes = await imageFile.readAsBytes();
    img.Image? originalImg = img.decodeImage(bytes);
    if (originalImg == null) return "";

    img.Image resizedImg = img.copyResize(originalImg, width: 640, height: 640);

    // 2. Prepare Float32 input matrix [1, 640, 640, 3]
    var input = List.generate(
      1,
      (_) => List.generate(
        640,
        (y) => List.generate(640, (x) {
          var pixel = resizedImg.getPixel(x, y);
          return [pixel.r / 255.0, pixel.g / 255.0, pixel.b / 255.0];
        }),
      ),
    );

    // 3. Prepare output matrix shape [1, 59, 8400]
    var output = List.filled(1 * 59 * 8400, 0.0).reshape([1, 59, 8400]);

    // 4. Run TFLite inference
    _interpreter!.run(input, output);

    // 5. Parse bounding boxes and confidence scores
    List<Map<String, dynamic>> detections = [];
    double confidenceThreshold = 0.25;

    for (int i = 0; i < 8400; i++) {
      double xCenter = output[0][0][i]; // box center X
      double maxScore = 0.0;
      int bestClassId = -1;

      // Check class scores across 55 categories (index 4 to 58)
      for (int c = 0; c < 55; c++) {
        double score = output[0][4 + c][i];
        if (score > maxScore) {
          maxScore = score;
          bestClassId = c;
        }
      }

      if (maxScore >= confidenceThreshold && bestClassId != -1) {
        detections.add({
          'x': xCenter,
          'label': classNames[bestClassId],
          'score': maxScore,
        });
      }
    }

    // 6. Sort detected characters left-to-right along the X-axis to reconstruct word
    detections.sort((a, b) => (a['x'] as double).compareTo(b['x'] as double));

    // Combine distinct character labels
    String reconstructedWord = detections.map((d) => d['label']).join('-');
    return reconstructedWord;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("🏛️ Ancient Cypriot OCR")),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Container(
              height: 260,
              width: double.infinity,
              decoration: BoxDecoration(
                border: Border.all(color: Colors.grey.shade400),
                borderRadius: BorderRadius.circular(12),
              ),
              child: _selectedImage != null
                  ? Image.file(_selectedImage!, fit: BoxFit.contain)
                  : const Center(child: Text("Select an inscription image")),
            ),
            const SizedBox(height: 20),
            ElevatedButton.icon(
              onPressed: _pickAndProcessImage,
              icon: const Icon(Icons.photo_library),
              label: const Text("Pick Image & Recognize"),
            ),
            const SizedBox(height: 30),
            if (_isLoading) const CircularProgressIndicator(),
            if (_extractedText.isNotEmpty && !_isLoading)
              Card(
                color: Colors.amber.shade50,
                elevation: 3,
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    children: [
                      const Text(
                        "Reconstructed Word:",
                        style: TextStyle(fontSize: 14, color: Colors.black54),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        _extractedText,
                        style: const TextStyle(
                          fontSize: 26,
                          fontWeight: FontWeight.bold,
                          color: Colors.amber,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
