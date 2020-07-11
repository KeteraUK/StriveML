import XCTest
@testable import StriveML

final class StriveMLTests: XCTestCase {
    
    var sut: StriveML!
    
    override func setUp() {
        super.setUp()
        sut = StriveML()
    }

    override func tearDown() {
        sut = nil
        super.tearDown()
    }
    
    func test_predictionAccuracy_givenDataset() {
        let dataset: [[Double]] = [[1, 140], [2, 150], [3, 170], [4, 180]]
        sut.set(x: 5, data: dataset)
        var result = try! sut.predict()
        XCTAssertEqual(result.y, 195)
        XCTAssertEqual(result.pcc, 0.98995)
        
        sut.set(x: 8, data: dataset)
        result = try! sut.predict()
        XCTAssertEqual(result.y, 237)
        XCTAssertEqual(result.pcc, 0.98995)
    }

    func test_dataset_validation() {
        
        var dataset = [[Double]]()

        dataset = [[1, 140], [2, 150]]
        sut.set(x: 5, data: dataset)
        XCTAssertThrowsError(try sut.predict()) { error in
            XCTAssertEqual(error as? MLError, MLError.data(message: "This dataset is too limited, provide at least 4 observations."))
        }

        dataset = [[1, 140], [2], [3, 170], [4, 180]]
        sut.set(x: 5, data: dataset)
        XCTAssertThrowsError(try sut.predict()) { error in
            XCTAssertEqual(error as? MLError, MLError.data(message: "Number of x and y in observations is unequal."))
        }

        dataset = [[1, 140], [], [3, 170], [4, 180]]
        sut.set(x: 5, data: dataset)
        XCTAssertThrowsError(try sut.predict()) { error in
            XCTAssertEqual(error as? MLError, MLError.data(message: "Number of x and y in observations is unequal."))
        }
    }
}
