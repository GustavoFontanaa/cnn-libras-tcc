import React, { useEffect, useState, useRef } from "react";
import { View, Text, TouchableOpacity, StyleSheet } from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";

export default function App() {
  const CLASSES_LIBRAS = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
  ];

  const [permission, requestPermission] = useCameraPermissions();
  const [ws, setWs] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const cameraRef = useRef(null);

  useEffect(() => {
    const socket = new WebSocket("ws://192.168.0.102:8080");
    socket.onopen = () => console.log("✅ Conectado ao WebSocket");
    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.prediction) {
        setPrediction(data.prediction);
      } else if (data.error) {
        console.error("Erro do servidor:", data.error);
      }
    };
    socket.onerror = (error) => console.error("Erro no WebSocket:", error);
    socket.onclose = () => console.log("❌ WebSocket desconectado");

    setWs(socket);

    return () => {
      socket.close();
    };
  }, []);

  if (!permission) {
    return <View />;
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={{ textAlign: "center" }}>
          Precisamos da permissão da câmera
        </Text>
        <TouchableOpacity onPress={requestPermission} style={styles.button}>
          <Text style={styles.text}>Conceder permissão</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const interpretarPrevisao = (predictions) => {
    console.log("Previsões recebidas:", predictions);

    if (!Array.isArray(predictions) || predictions.length === 0) {
      console.error("Erro: O array de previsões está vazio ou inválido.");
      return "Erro";
    }

    const valoresPrevisao = predictions[0]; // Pega apenas o primeiro array interno
    const indexMaior = valoresPrevisao.indexOf(Math.max(...valoresPrevisao));

    if (indexMaior === -1) {
      console.error("Erro: Não foi possível encontrar um índice válido.");
      return "Erro";
    }

    return CLASSES_LIBRAS[indexMaior] || "Desconhecido";
  };

  const captureAndSend = async () => {
    if (cameraRef.current) {
      const photo = await cameraRef.current.takePictureAsync({ base64: true });

      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ image: photo.base64 }));
        console.log("📸 Imagem enviada ao servidor!");
      } else {
        console.error("❌ WebSocket não está conectado.");
      }
    }
  };

  return (
    <View style={styles.container}>
      <CameraView style={styles.camera} ref={cameraRef} facing="back">
        <View style={styles.overlay}>
          <TouchableOpacity onPress={captureAndSend} style={styles.button}>
            <Text style={styles.text}>Capturar e Enviar</Text>
          </TouchableOpacity>
        </View>
      </CameraView>
      {prediction && (
        <View style={styles.predictionContainer}>
          <Text style={styles.predictionTitle}>📢 Sinal Detectado:</Text>
          <Text style={styles.predictionText}>
            {interpretarPrevisao(prediction)}
          </Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: "center" },
  camera: { flex: 1 },
  overlay: {
    flex: 1,
    justifyContent: "flex-end",
    alignItems: "center",
    marginBottom: 30,
  },
  button: { backgroundColor: "blue", padding: 10, borderRadius: 5 },
  text: { color: "white", fontWeight: "bold" },
  result: {
    padding: 20,
    backgroundColor: "rgba(0,0,0,0.5)",
    position: "absolute",
    bottom: 20,
    alignSelf: "center",
  },

  predictionContainer: {
    backgroundColor: "#4CAF50",
    padding: 20,
    borderRadius: 10,
    marginTop: 20,
    alignItems: "center",
    width: "80%",
  },
  predictionTitle: {
    fontSize: 18,
    fontWeight: "bold",
    color: "white",
  },
  predictionText: {
    fontSize: 24,
    color: "white",
    fontWeight: "bold",
  },
});
