package kia.demo.ml.onnx.service;


import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.stereotype.Service;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

@Service
public class OnnxModelService {
    private OrtEnvironment env;
    private OrtSession session;
    private double accuracy;
    private Map<Integer, String> classMapping = new HashMap<>();

    @Autowired
    private ResourceLoader resourceLoader;

    public OnnxModelService() throws OrtException, IOException {
        // Initialize ONNX Runtime environment
        env = OrtEnvironment.getEnvironment();
        // Load the model from the classpath
        Resource resource = resourceLoader.getResource("classpath:iris_classifier.onnx");
        Path modelPath = resource.getFile().toPath();
        session = env.createSession(modelPath.toString());

        // Load class mapping from JSON
        Resource mappingResource = resourceLoader.getResource("classpath:class_mapping.json");
        ObjectMapper objectMapper = new ObjectMapper();
        classMapping = objectMapper.readValue(mappingResource.getFile(), HashMap.class);

        // Hardcode accuracy for demonstration
        this.accuracy = 1.0; // Replace with actual calculated accuracy if desired
    }

    public float[] predict(float[][] inputData) throws OrtException {
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputData);
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("input", inputTensor);

        //main part of model in session
        OrtSession.Result outputs = session.run(inputs);

        Optional<OnnxValue> optionalValue = outputs.get("output");
        OnnxTensor predictionsTensor = (OnnxTensor) optionalValue.get();
        float[] predictions = (float[]) predictionsTensor.getValue();

        // Clean up
        inputTensor.close();
        outputs.close();
        return predictions;
    }
}
