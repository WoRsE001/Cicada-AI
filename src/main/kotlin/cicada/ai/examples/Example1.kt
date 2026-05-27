package cicada.ai.examples

import cicada.ai.Activation
import cicada.ai.LayerConfig
import cicada.ai.NeuralNetwork

// SCWGxD regrets everything he did. 27.05.2026 4:56.
fun main() {
    val model = NeuralNetwork(1, listOf(
        LayerConfig(1, Activation.LINEAR),
    ))

    val dataSet = listOf(
        Pair(doubleArrayOf(0.0), doubleArrayOf(1.0)),
        Pair(doubleArrayOf(0.1), doubleArrayOf(0.9)),
        Pair(doubleArrayOf(0.2), doubleArrayOf(0.8)),
        Pair(doubleArrayOf(0.3), doubleArrayOf(0.7)),
        Pair(doubleArrayOf(0.4), doubleArrayOf(0.6)),
        Pair(doubleArrayOf(0.5), doubleArrayOf(0.5)),
        Pair(doubleArrayOf(0.6), doubleArrayOf(0.4)),
        Pair(doubleArrayOf(0.7), doubleArrayOf(0.3)),
        Pair(doubleArrayOf(0.8), doubleArrayOf(0.2)),
        Pair(doubleArrayOf(0.9), doubleArrayOf(0.1)),
        Pair(doubleArrayOf(1.0), doubleArrayOf(0.0)),
    )

    model.train(dataSet, 10000)

    while (true) {
        print("Enter number: ")

        val input = readlnOrNull()?.toDoubleOrNull() ?: continue

        println("%.4f".format(model.predict(doubleArrayOf(input))[0]))
    }
}