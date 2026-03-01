import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const TyphoonDayoffApp());
}

class TyphoonDayoffApp extends StatelessWidget {
  const TyphoonDayoffApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '颱風假預測',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
        useMaterial3: true,
      ),
      home: const PredictionPage(),
    );
  }
}

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  /// IMPORTANT for beginners:
  ///
  /// 1) Android emulator uses 10.0.2.2 to reach your local machine.
  /// 2) iOS simulator can use localhost.
  /// 3) Real phone must use your computer LAN IP (e.g., 192.168.x.x).
  static const String baseUrl = String.fromEnvironment(
    'API_BASE_URL',
    defaultValue: 'http://10.0.2.2:5000',
  );

  bool loadingLocations = true;
  bool loadingPrediction = false;
  String? errorText;
  List<String> locations = [];
  String? selectedLocation;
  PredictionResult? result;

  @override
  void initState() {
    super.initState();
    _loadLocations();
  }

  Future<void> _loadLocations() async {
    setState(() {
      loadingLocations = true;
      errorText = null;
    });

    try {
      final uri = Uri.parse('$baseUrl/app/locations');
      final response = await http.get(uri);
      final body = jsonDecode(response.body) as Map<String, dynamic>;

      if (response.statusCode != 200 || body['ok'] != true) {
        throw Exception(body['error'] ?? '無法取得測站清單');
      }

      final rawLocations = (body['locations'] as List<dynamic>)
          .map((e) => e.toString())
          .toList();

      setState(() {
        locations = rawLocations;
        selectedLocation = rawLocations.isNotEmpty ? rawLocations.first : null;
      });
    } catch (e) {
      setState(() {
        errorText = '讀取測站失敗：$e';
      });
    } finally {
      setState(() {
        loadingLocations = false;
      });
    }
  }

  Future<void> _predict() async {
    if (selectedLocation == null) return;

    setState(() {
      loadingPrediction = true;
      errorText = null;
    });

    try {
      final uri = Uri.parse(
        '$baseUrl/app/predict?locationName=${Uri.encodeComponent(selectedLocation!)}',
      );
      final response = await http.get(uri);
      final body = jsonDecode(response.body) as Map<String, dynamic>;

      if (response.statusCode != 200 || body['ok'] != true) {
        throw Exception(body['error'] ?? '預測失敗');
      }

      setState(() {
        result = PredictionResult.fromJson(body);
      });
    } catch (e) {
      setState(() {
        errorText = '預測失敗：$e';
      });
    } finally {
      setState(() {
        loadingPrediction = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('颱風假預測 App')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: loadingLocations
            ? const Center(child: CircularProgressIndicator())
            : Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('請先選擇測站：'),
                  const SizedBox(height: 8),
                  DropdownButtonFormField<String>(
                    value: selectedLocation,
                    items: locations
                        .map((loc) => DropdownMenuItem(value: loc, child: Text(loc)))
                        .toList(),
                    onChanged: (value) {
                      setState(() {
                        selectedLocation = value;
                      });
                    },
                    decoration: const InputDecoration(border: OutlineInputBorder()),
                  ),
                  const SizedBox(height: 12),
                  SizedBox(
                    width: double.infinity,
                    child: FilledButton(
                      onPressed: loadingPrediction ? null : _predict,
                      child: loadingPrediction
                          ? const SizedBox(
                              width: 16,
                              height: 16,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          : const Text('查詢最新預測'),
                    ),
                  ),
                  if (errorText != null) ...[
                    const SizedBox(height: 12),
                    Text(errorText!, style: const TextStyle(color: Colors.red)),
                  ],
                  if (result != null) ...[
                    const SizedBox(height: 16),
                    Card(
                      child: Padding(
                        padding: const EdgeInsets.all(12),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text('測站：${result!.locationName}'),
                            Text('氣溫：${result!.temperatureC} °C'),
                            Text('雨量：${result!.rainfallMm} mm'),
                            Text('天氣：${result!.description}'),
                            Text('放假機率：${result!.dayoffProbability}%'),
                            const SizedBox(height: 8),
                            Text('建議：${result!.advice}'),
                          ],
                        ),
                      ),
                    ),
                  ],
                ],
              ),
      ),
    );
  }
}

class PredictionResult {
  final String locationName;
  final String temperatureC;
  final String rainfallMm;
  final String description;
  final double dayoffProbability;
  final String advice;

  PredictionResult({
    required this.locationName,
    required this.temperatureC,
    required this.rainfallMm,
    required this.description,
    required this.dayoffProbability,
    required this.advice,
  });

  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    final weather = json['weather'] as Map<String, dynamic>;
    return PredictionResult(
      locationName: json['locationName'].toString(),
      temperatureC: weather['temperature_c'].toString(),
      rainfallMm: weather['rainfall_mm'].toString(),
      description: weather['description'].toString(),
      dayoffProbability: (json['dayoff_probability'] as num).toDouble(),
      advice: json['advice'].toString(),
    );
  }
}
