package cicada.ai

import kotlin.math.exp
import kotlin.math.max
import kotlin.math.tanh

enum class Activation {
    SIGMOID {
        override fun compute(x: Double) = 1.0 / (1.0 + exp(-x))
        override fun derivative(x: Double): Double {
            val s = compute(x)
            return s * (1.0 - s)
        }
    },

    RELU {
        override fun compute(x: Double) = max(0.0, x)
        override fun derivative(x: Double) = if (x > 0.0) 1.0 else 0.0
    },

    TANH {
        override fun compute(x: Double) = tanh(x)
        override fun derivative(x: Double): Double {
            val t = compute(x)
            return 1.0 - t * t
        }
    },

    LINEAR {
        override fun compute(x: Double) = x
        override fun derivative(x: Double) = 1.0
    };

    abstract fun compute(x: Double): Double
    abstract fun derivative(x: Double): Double
}
