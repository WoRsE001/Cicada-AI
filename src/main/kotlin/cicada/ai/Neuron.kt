package cicada.ai

import kotlin.math.sqrt
import kotlin.random.Random

// SCWGxD regrets everything he did. 25.05.2026 6:53.
class Neuron(
    val inputCount: Int,
    val activation: Activation
) {
    var weights: DoubleArray = DoubleArray(inputCount) {
        Random.nextDouble(-1.0, 1.0) * sqrt(2.0 / inputCount)
    }

    var bias: Double = 0.0
    var lastInput: DoubleArray = DoubleArray(inputCount)
    var lastNet: Double = 0.0
    var lastOutput: Double = 0.0
    var delta: Double = 0.0

    fun forward(inputs: DoubleArray): Double {
        require(inputs.size == inputCount) {
            "Expected $inputCount inputs, got ${inputs.size}"
        }
        lastInput = inputs.copyOf()
        lastNet = inputs.indices.sumOf { i -> inputs[i] * weights[i] } + bias
        lastOutput = activation.compute(lastNet)
        return lastOutput
    }
}