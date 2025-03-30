const tf = require('@tensorflow/tfjs-node');
const WebSocket = require('ws');
const express = require('express');
const cors = require('cors');

const app = express();
app.use(cors());

const wss = new WebSocket.Server({ port: 8080 });
let model = null;

// Função para carregar o modelo
const loadModel = async () => {
  try {
    console.log('🔄 Carregando modelo...');
    model = await tf.loadLayersModel('file://model/model.json');
    console.log('✅ Modelo carregado com sucesso!');
  } catch (error) {
    console.error('❌ Erro ao carregar o modelo:', error);
  }
};

// Inicia o carregamento do modelo
loadModel();

// WebSocket para comunicação
wss.on('connection', (ws) => {
  console.log('🔌 Cliente conectado ao WebSocket');

  ws.on('message', async (message) => {
    try {
      const data = JSON.parse(message);
      if (data.image) {
        console.log('📸 Recebida uma imagem para processamento');

        // Convertendo a imagem em tensor (aqui você pode precisar ajustar)
        const buffer = Buffer.from(data.image, 'base64');
        const decodedImage = tf.node.decodeImage(buffer);
        const resizedImage = decodedImage.resizeNearestNeighbor([64, 64]).expandDims(0).toFloat().div(tf.scalar(255));

        // Fazendo a previsão
        const predictions = model.predict(resizedImage);
        console.log("Saída do modelo:", predictions.arraySync());

        const result = predictions.arraySync();

        // Enviar resultado de volta
        ws.send(JSON.stringify({ prediction: result }));
      }
    } catch (error) {
      console.error('❌ Erro ao processar a imagem:', error);
      ws.send(JSON.stringify({ error: 'Erro ao processar a imagem' }));
    }
  });

  ws.on('close', () => {
    console.log('❌ Cliente desconectado');
  });
});

// Servidor HTTP opcional
app.get('/', (req, res) => {
  res.send('Servidor TensorFlow WebSocket rodando...');
});

app.listen(3000, () => {
  console.log('🚀 Servidor HTTP rodando em http://localhost:3000');
});
