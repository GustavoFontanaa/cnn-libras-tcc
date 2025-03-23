import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View, Button } from 'react-native';
import { useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';

export default function App() {
  const [isTfReady, setIsTfReady] = useState(false);

  useEffect(() => {
    const loadTensorFlow = async () => {
      try {
        console.log("Carregando TensorFlow...");
        await tf.ready();
        console.log("TensorFlow pronto!");
        setIsTfReady(true);
      } catch (error) {
        console.error("Erro ao carregar TensorFlow:", error);
      }
    };

    loadTensorFlow();
  }, []);

  const testTensorFlow = async () => {
    if (!isTfReady) {
      console.log("TensorFlow ainda não está pronto.");
      return;
    }
  
    console.log("Executando TensorFlow...");
  
    try {
      const tensor = tf.tensor([1, 2, 3, 4], [2, 2]);
      console.log("Tensor criado:", tensor);
  
      const result = tensor.sum();
      console.log("Resultado da soma:", result);
  
      const resultArray = await result.data();
      console.log("Resultado final:", resultArray);
  
    } catch (error) {
      console.error("Erro ao executar TensorFlow:", error);
    }
  };
  

  return (
    <View style={styles.container}>
      <Text style={styles.text}>
        {isTfReady ? "TensorFlow pronto!" : "Carregando TensorFlow..."}
      </Text>
      <Button title="Testar TensorFlow" onPress={testTensorFlow} disabled={!isTfReady} />
      <StatusBar style="auto" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'blue',
    alignItems: 'center',
    justifyContent: 'center',
  },
  text: {
    color: 'white',
    fontSize: 20,
    marginBottom: 20,
  },
});
