import cv2
import json
from pipeline.preprocessing_pipeline import PreprocessingPipeline

def main():

    image_path = "input.jpg"
    image = cv2.imread(image_path)

    pipeline = PreprocessingPipeline(output_dir="output")

    final_image, results = pipeline.run(image)

    # Save final image
    cv2.imwrite("output/final_output.jpg", final_image)

    # Save JSON report
    with open("output/report.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Processing completed.")

if __name__ == "__main__":
    main()