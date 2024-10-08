package kia.demo.ml.onnx.controller;

import ai.onnxruntime.OrtException;
import kia.demo.ml.onnx.service.OnnxModelService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/predict")
public class IrisPredictionController {

    @Autowired
    private OnnxModelService onnxModelService;

    @PostMapping
    public float[] predict(@RequestBody float[][] inputData) {
        try {
            return onnxModelService.predict(inputData);
        } catch (OrtException e) {
            throw new RuntimeException("Inference failed", e);
        }
    }
}