// On-screen text recognition for YouScoop keyframes with Apple's Vision
// framework. Runs locally. Reads image paths from argv, writes one JSON array:
// [{"path": ..., "lines": [{"text", "confidence", "x", "y", "w", "h"}]}]
// Coordinates are normalized with the origin at the top left.
import Foundation
import Vision

struct Line: Encodable { let text: String; let confidence: Float; let x, y, w, h: Double }
struct Page: Encodable { let path: String; let lines: [Line]; let error: String? }

var pages: [Page] = []
for path in CommandLine.arguments.dropFirst() {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    request.automaticallyDetectsLanguage = true
    do {
        try VNImageRequestHandler(url: URL(fileURLWithPath: path)).perform([request])
        let lines = (request.results ?? []).compactMap { observation -> Line? in
            guard let best = observation.topCandidates(1).first else { return nil }
            let box = observation.boundingBox
            return Line(text: best.string, confidence: best.confidence,
                        x: box.minX, y: 1 - box.maxY, w: box.width, h: box.height)
        }
        pages.append(Page(path: path, lines: lines, error: nil))
    } catch {
        pages.append(Page(path: path, lines: [], error: error.localizedDescription))
    }
}
let encoder = JSONEncoder()
FileHandle.standardOutput.write(try! encoder.encode(pages))
