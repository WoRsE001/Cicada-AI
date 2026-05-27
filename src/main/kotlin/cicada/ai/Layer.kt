package cicada.ai

// SCWGxD regrets everything he did. 25.05.2026 6:53.
class Layer(
    val neuronCount: Int,
    val inputCount: Int,
    val activation: Activation
) {
    val neurons: List<Neuron> = List(neuronCount) { Neuron(inputCount, activation) }

    fun forward(inputs: DoubleArray): DoubleArray =
        DoubleArray(neuronCount) { i -> neurons[i].forward(inputs) }

    override fun toString(): String =
        "Layer(neurons=$neuronCount, inputs=$inputCount, activation=${activation.name})"
}