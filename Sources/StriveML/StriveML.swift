import Foundation

public typealias NestedDouble = [[Double]]

public enum MLError: Error, Equatable {
    case data(message: String)
}

public struct MLResult {
    public let pcc, x, y: Double

    public init(pcc: Double, x: Double, y: Double) {
        self.pcc = pcc
        self.x = x
        self.y = y
    }
}

open class StriveML {
    // MARK: - Properties
    public final let minArguments = 4

    public private(set) var x = Double()
    public private(set) var xV = [Double]()
    public private(set) var yV = [Double]()
    public private(set) var data = NestedDouble()
    public private(set) var count = Int()

    public init() {}

    /// Class initializer
    /// - Parameters:
    ///   - x: Given value `x` to predict `y`
    ///   - data: Data set of observations
    public init(x: Double, data: NestedDouble) {
        self.set(x: x, data: data)
    }

    /// Override instance properties
    /// - Parameters:
    ///   - x: Given value `x` to predict `y`
    ///   - data: Data set of observations
    public func set(x: Double, data: NestedDouble? = nil) {
        self.x = x

        if let data = data {
            self.count = data.count
            self.xV = []
            self.yV = []
            self.data = data
                .map {
                    if $0.indices.contains(0), $0.indices.contains(1) {
                        self.xV.append($0[0])
                        self.yV.append($0[1])
                    }
                    return $0
                }
        }
    }

    /// Sum vector values
    /// - Parameter vector: Vector values to be summed
    private func sum(vector: [Double]) -> Double {
        return vector
            .reduce(0, +)
    }

    /// Sum vector² values
    /// - Parameter vector: Squared vector values to be summed
    private func squareSum(vector: [Double]) -> Double {
        return vector
            .map {
                power(base: $0, exponent: 2)
            }
            .reduce(0, +)
    }

    /// Sum of x*y
    /// - Parameter data: Data set of x and y values
    private func sumXY(data: NestedDouble) -> Double {
        return data
            .map {
                $0[0] * $0[1]
            }
            .reduce(0, +)
    }

    /// Apply power to a base value
    /// - Parameters:
    ///   - v: Base value `v`
    ///   - exponent: The `expontent` or the number of times `v` should be multiplied by itself
    private func power(base v: Double, exponent: Double) -> Double {
        return pow(v, exponent)
    }

    /// Described data set spread
    /// - Parameter v: Variability given axis `v`
    private func variability(v: String) -> Double {
        let squareSum = [
            "x": { self.squareSum(vector: self.xV) },
            "y": { self.squareSum(vector: self.yV) }
        ]

        let sum = [
            "x": { self.sum(vector: self.xV) },
            "y": { self.sum(vector: self.yV) }
        ]

        let vSquare = squareSum[v]!
        let vSum = sum[v]!

        return (vSquare() / Double(self.count))
            - self.power(base: vSum() / Double(self.count), exponent: 2)
    }

    /// Intercept such that the line passes through the center of mass (x, y) of the data points.
    /// - Parameter b: Slope `b` to find intercept
    private func intercept(b: Double) -> Double {
        return (self.sum(vector: self.yV) / Double(self.count)) -
            (b * (self.sum(vector: self.xV) / Double(self.count)))
    }

    /// Regression coefficient
    private func slope() -> Double {
        return ((self.sumXY(data: self.data) / Double(self.count)) -
            ((self.sum(vector: self.xV) / Double(self.count)) *
                (self.sum(vector: self.yV) / Double(self.count)))) /
            self.variability(v: "x")
    }

    /// Pearson's correlation coefficient - measure of the linear correlation between two variables X and Y
    /// - Parameter b: Regression coefficient `b`
    private func pcc(b: Double) -> Double {
        return b * (sqrt(self.variability(v: "x"))
            / sqrt(self.variability(v: "y")))
    }

    /// Create a model to fit data
    /// - Parameters:
    ///   - a: Intercept `a`
    ///   - b: Regression coefficient `b`
    private func createLinearModel(a: Double, b: Double) -> (_ x: Double) -> Double {
        return { (x: Double) -> Double in
            a + b * x
        }
    }

    private func round_(number: Double) -> Double {
        return round(100000 * number) / 100000
    }

    /// Create a linear regression model to predict value `y` given `x`
    public func predict() throws -> MLResult {
        /// Validate the number of arguments
        guard self.count >= self.minArguments else {
            throw MLError.data(message: "This dataset is too limited, provide at least 4 observations.")
        }

        /// Ensure x & y count is equal
        guard self.count == self.xV.count else {
            throw MLError.data(message: "Number of x and y in observations is unequal.")
        }

        let b = self.round_(number: self.slope())
        let a = self.round_(number: self.intercept(b: b))
        let model = self.createLinearModel(a: a, b: b)
        let y = self.round_(number: model(self.x))

        return MLResult(
            pcc: self.round_(number: self.pcc(b: b)),
            x: self.x,
            y: y
        )
    }
}
