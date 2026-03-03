import cv2
import json
from pipeline.preprocessing_pipeline import PreprocessingPipeline
from utils.image_utils import make_json_safe

def main():
    print("loading input image")
    image_path = "/content/drive/MyDrive/img_processing/imgs/t3_sku.jpg"
    image = cv2.imread(image_path)
    
    pipeline = PreprocessingPipeline(output_dir="output")

    final_image, results = pipeline.run(image)
    print(results)
    # Save final image
    cv2.imwrite("output/final_output.jpg", final_image)

    # Save JSON report
    safe_results = make_json_safe(results)

    with open("output/report.json", "w") as f:
      json.dump(safe_results, f, indent=4)

    print("Processing completed.")

if __name__ == "__main__":
    main()