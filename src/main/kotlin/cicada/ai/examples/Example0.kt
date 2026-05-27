package cicada.ai.examples

import cicada.ai.Activation
import cicada.ai.LayerConfig
import cicada.ai.NeuralNetwork
import kotlin.math.roundToInt

// SCWGxD regrets everything he did. 27.05.2026 2:26.
fun main() {
    val model = NeuralNetwork(2, listOf(
        LayerConfig(4, Activation.RELU),
        LayerConfig(4, Activation.RELU),
        LayerConfig(1, Activation.RELU),
    ))

    val dataSet = listOf(
        Pair(doubleArrayOf(0.0, 0.0), doubleArrayOf(0.0)),
        Pair(doubleArrayOf(0.0, 1.0), doubleArrayOf(1.0)),
        Pair(doubleArrayOf(1.0, 0.0), doubleArrayOf(1.0)),
        Pair(doubleArrayOf(1.0, 1.0), doubleArrayOf(1.0)),
    )

    model.train(dataSet, 1000)

    while (true) {

        print("Enter 2 numbers: ")

        val inputString = readlnOrNull()
            ?.trim()
            ?.split(",")
            ?: continue

        val inputInt = inputString.mapNotNull {
            it.trim().toIntOrNull()
        }

        if (inputInt.size != model.inputCount) {
            println("Expected ${model.inputCount} inputs, got ${inputInt.size}")
            continue
        }

        val input = DoubleArray(inputInt.size) {
            inputInt[it].toDouble()
        }

        println(model.predict(input)[0].roundToInt())
    }
}