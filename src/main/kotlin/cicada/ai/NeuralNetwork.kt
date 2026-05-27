package cicada.ai

import kotlinx.serialization.json.*

// SCWGxD regrets everything he did. 25.05.2026 6:52.
open class NeuralNetwork(
    val inputCount: Int,
    layerConfigs: List<LayerConfig>
) {
    val layerSizes = listOf(inputCount) + layerConfigs.map { it.neuronCount }
    val layers = layerConfigs.mapIndexed { index, config ->
        Layer(config.neuronCount, layerSizes[index], config.activation)
    }
    var avgLoss = -1.0

    fun predict(inputs: DoubleArray): DoubleArray {
        require(inputs.size == inputCount) {
            "Expected $inputCount inputs, got ${inputs.size}"
        }
        var signal = inputs
        for (layer in layers) {
            signal = layer.forward(signal)
        }
        return signal
    }

    private fun trainStep(inputs: DoubleArray, targets: DoubleArray, learningRate: Double): Double {
        val outputCount = layers.last().neuronCount

        require(targets.size == outputCount) {
            "Expected $outputCount target values, got ${targets.size}"
        }

        val output = predict(inputs)

        val outputLayer = layers.last()
        for (i in 0 until outputCount) {
            val n = outputLayer.neurons[i]
            val error = targets[i] - n.lastOutput
            n.delta = error * n.activation.derivative(n.lastNet)
        }

        for (li in layers.size - 2 downTo 0) {
            val currentLayer = layers[li]
            val nextLayer    = layers[li + 1]

            for (ci in currentLayer.neurons.indices) {
                val neuron = currentLayer.neurons[ci]
                val weightedDeltaSum = nextLayer.neurons.sumOf { nj -> nj.delta * nj.weights[ci] }
                neuron.delta = weightedDeltaSum * neuron.activation.derivative(neuron.lastNet)
            }
        }

        for (li in layers.indices) {
            val layer = layers[li]
            for (neuron in layer.neurons) {
                for (wi in neuron.weights.indices) {
                    neuron.weights[wi] += learningRate * neuron.delta * neuron.lastInput[wi]
                }
                neuron.bias += learningRate * neuron.delta
            }
        }

        return output.indices.sumOf { i ->
            val e = targets[i] - output[i]; e * e
        } / outputCount
    }

    fun train(
        dataset: List<Pair<DoubleArray, DoubleArray>>,
        epochs: Int,
        learningRate: Double = 0.1
    ) {
        require(dataset.isNotEmpty()) { "Dataset is empty" }

        repeat (epochs) {
            avgLoss = dataset.sumOf { (x, y) ->
                trainStep(x, y, learningRate)
            } / dataset.size
        }
    }

    val asJson: JsonObject
        get() = buildJsonObject {

            putJsonObject("info") {
                put("input", inputCount)

                putJsonArray("layers") {
                    layerSizes.drop(1).forEach { add(it) }
                }

                put("avgLoss", avgLoss)
            }

            putJsonArray("layers") {
                for (layer in layers) {
                    addJsonObject {
                        put("neuronCount", layer.neuronCount)
                        put("activation", layer.neurons.first().activation.name)

                        putJsonArray("neurons") {
                            for (neuron in layer.neurons) {
                                addJsonObject {
                                    put("bias", neuron.bias)

                                    putJsonArray("weights") {
                                        neuron.weights.forEach { add(it) }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

    companion object {

        fun fromJson(json: JsonObject): NeuralNetwork {

            val info = json["info"]?.jsonObject

            val inputCount = info
                ?.get("input")
                ?.jsonPrimitive
                ?.int ?: 0

            val layersJson = json["layers"]
                ?.jsonArray ?: emptyList()

            val layerConfigs = layersJson.mapNotNull { layerElement ->

                val layerObject = layerElement.jsonObject

                val neuronCount = layerObject["neuronCount"]
                    ?.jsonPrimitive
                    ?.int ?: return@mapNotNull null

                val activation = layerObject["activation"]
                    ?.jsonPrimitive
                    ?.content
                    ?.let {
                        runCatching { Activation.valueOf(it) }.getOrNull()
                    }
                    ?: Activation.TANH

                LayerConfig(
                    neuronCount = neuronCount,
                    activation = activation
                )
            }

            val network = NeuralNetwork(inputCount, layerConfigs)

            info?.get("avgLoss")
                ?.jsonPrimitive
                ?.doubleOrNull
                ?.let {
                    network.avgLoss = it
                }

            for ((layerIndex, layerElement) in layersJson.withIndex()) {

                val layerObject = layerElement.jsonObject

                val neuronsJson = layerObject["neurons"]
                    ?.jsonArray ?: continue

                val layer = network.layers
                    .getOrNull(layerIndex)
                    ?: continue

                for ((neuronIndex, neuronElement) in neuronsJson.withIndex()) {

                    val neuronObject = neuronElement.jsonObject

                    val neuron = layer.neurons
                        .getOrNull(neuronIndex)
                        ?: continue

                    neuronObject["bias"]
                        ?.jsonPrimitive
                        ?.doubleOrNull
                        ?.let {
                            neuron.bias = it
                        }

                    neuronObject["weights"]
                        ?.jsonArray
                        ?.mapNotNull {
                            it.jsonPrimitive.doubleOrNull
                        }
                        ?.toDoubleArray()
                        ?.let {
                            neuron.weights = it
                        }
                }
            }

            return network
        }
    }
}

data class LayerConfig(
    val neuronCount: Int,
    val activation: Activation = Activation.TANH
)