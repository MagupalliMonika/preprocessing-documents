from models.sam_loader import load_sam_model
from techniques.document_boundary import DocumentBoundaryDetector
from techniques.perspective import PerspectiveCorrection
from pipeline.orientation_pipeline import correct_document_orientation
import cv2
from utils.image_utils import save_output
from pipeline.blur import BlurDetection
from pipeline.illumination import IlluminationCorrection
from pipeline.noise import NoiseReduction
from pipeline.uneven_illumination import UnevenIlluminationCorrection
from pipeline.resolution import ResolutionEnhancement
from pipeline.blank_page import BlankPageDetection
from pipeline.background_normalization import BackgroundNormalization
from pipeline.stroke_width_check import StrokeWidthConsistency


class PreprocessingPipeline:

    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        self.bg_normalizer = BackgroundNormalization()
        self.stroke_check = StrokeWidthConsistency(deviation_threshold=0.3)
        self.blank = BlankPageDetection(blank_threshold=0.02)
        self.resolution = ResolutionEnhancement(
                          text_height_threshold=30,
                          scale_factor=2.0
                      )
        self.uneven_illum = UnevenIlluminationCorrection(alpha=0.8)
        self.noise = NoiseReduction(sigma_threshold=0.02)
        self.blur = BlurDetection(threshold=0.65)
        self.mask_generator = load_sam_model()
        self.illumination = IlluminationCorrection()
        self.boundary = DocumentBoundaryDetector(
            mask_generator=self.mask_generator,
            output_dir=output_dir
        )

        self.perspective = PerspectiveCorrection(output_dir=output_dir)
    def run(self, image):

      results = {"techniques": {}}

      print("\n================ DOCUMENT PREPROCESSING PIPELINE ================\n")

      # ==========================================================
      # Technique 1: Blank Page Detection
      # ==========================================================
      print("Technique 1: Blank Page Detection")

      blank_result = self.blank.detect(image)

      results["techniques"]["blank_page"] = blank_result

      print(f"   → Foreground Ratio : {blank_result['foreground_ratio']:.5f}")
      print(f"   → Threshold        : {blank_result['threshold']}")
      print(f"   → Is Blank         : {blank_result['needed']}")

      if blank_result["needed"]:
          print("  Processing Stopped: Blank Page Detected\n")
          return image, results

      print(" Page Contains Content\n")


      # ==========================================================
      # Technique 2: Blur Detection
      # ==========================================================
      print(" Technique 2: Blur Detection")

      blur_result = self.blur.detect(image)
      results["techniques"]["blur_detection"] = blur_result

      print(f"   → Blur Score     : {blur_result['score']:.4f}")
      print(f"   → Threshold      : {blur_result['threshold']}")
      print(f"   → Is Blurry      : {blur_result['needed']}")

      if blur_result["needed"]:
          print("Processing Stopped: Image is too blurry.\n")
          return image, results

      print("Passed Blur Check\n")

      # ==========================================================
      # Technique 3: Orientation Correction & 4 skew correction
      # ==========================================================
      print("Technique 3: Orientation Correction & 4 skew correction")

      image, orientation_results = correct_document_orientation(image)
      orientation_data = orientation_results["orientation_correction"]

      results["techniques"]["orientation"] = orientation_data

      print(f"   → Rotation Applied : {orientation_data.get('rotation_applied')}")
      print(f"   → Skew Corrected   : {orientation_data.get('skew_corrected')}")
      print(f"   → Final Angle      : {orientation_data.get('final_angle')}")
      print("   Orientation Completed\n")

      # ==========================================================
      # Technique 5: Document Boundary Detection
      # ==========================================================
      print(" Technique 5: Document Boundary Detection")

      detect_boundary = self.boundary.detect(image)
      apply_boundary = self.boundary.apply(image, detect_boundary)

      results["techniques"]["document_boundary"] = {
          **detect_boundary,
          "output_path": apply_boundary["output_path"]
      }

      print(f"   → Document Found : {detect_boundary['needed']}")
      print(f"   → Confidence     : {detect_boundary.get('confidence', 'N/A')}")
      print(f"   → Output Saved   : {apply_boundary['output_path']}")

      if not detect_boundary["needed"]:
          print("    Processing Stopped: No document detected.\n")

          results["techniques"]["perspective"] = {
              "needed": False,
              "reason": "No document detected",
              "output_path": None
          }

          return image, results

      print(" Boundary Detection Completed\n")

      # ==========================================================
      # Technique 6: Perspective Correction
      # ==========================================================
      print(" Technique 6: Perspective Correction")

      detect_persp = self.perspective.detect(image, detect_boundary)

      if detect_persp["needed"]:
          apply_persp = self.perspective.apply(image, detect_boundary)
          image = apply_persp["processed_image"]
          output_path = apply_persp["output_path"]

          print("   → Perspective Needed : True")
          print(f"   → Output Saved       : {output_path}")
          print("   Perspective Applied\n")

      else:
          corners = detect_boundary["corners"]
          x, y, w, h = cv2.boundingRect(corners.astype("int32"))
          cropped = image[y:y+h, x:x+w]

          output_path = save_output(
              cropped,
              self.output_dir,
              "document_crop",
              "applied"
          )

          image = cropped

          print("   → Perspective Needed : False")
          print("   → Applied Fallback   : Bounding Box Crop")
          print(f"   → Output Saved       : {output_path}")
          print("    Cropping Applied\n")

      results["techniques"]["perspective"] = {
          **detect_persp,
          "output_path": output_path
      }

      # ==========================================================
      #  Technique 7: Low Resolution Detection & Enhancement
      # ==========================================================
      print(" Technique 7: Low Resolution Detection & Enhancement")

      detect_res = self.resolution.detect(image)

      results["techniques"]["resolution_enhancement"] = detect_res

      print(f"   → Avg Text Height : {detect_res['avg_text_height']:.2f} px")
      print(f"   → Threshold       : {detect_res['threshold']} px")
      print(f"   → Needs Upscale   : {detect_res['needed']}")

      if detect_res["needed"]:
          image = self.resolution.apply(image)

          output_path = save_output(
              image,
              self.output_dir,
              "resolution_upscaled",
              "applied"
          )

          results["techniques"]["resolution_enhancement"]["output_path"] = output_path

          print(f"   → Scale Factor    : {detect_res['scale_factor']}")
          print(f"   → Output Saved    : {output_path}")
          print("   Image Upscaled for Better OCR\n")

      else:
          results["techniques"]["resolution_enhancement"]["output_path"] = None
          print("   Resolution Sufficient — No Upscaling Needed\n")
        
      # ==========================================================
      #  Technique 8: Uneven Illumination Correction
      # ==========================================================
      print(" Technique 8: Uneven Illumination Correction")

      image = self.uneven_illum.apply(image)
      output_path = save_output(
          image,
          self.output_dir,
          "uneven_illumination",
          "applied"
      )
      results["techniques"]["uneven_illumination"] = {
          "applied": True,
          "alpha": self.uneven_illum.alpha,
          "blur_kernel": self.uneven_illum.blur_kernel,
          "output_path": output_path
      }

      print(f"   → Alpha            : {self.uneven_illum.alpha}")
      print(f"   → Blur Kernel      : {self.uneven_illum.blur_kernel}")
      print(f"   → Output Saved     : {output_path}")
      print("Uneven Illumination Corrected\n")


      # ==========================================================
      #  Technique 9: Background Color Normalization
      # ==========================================================
      print("Technique 9: Background Color Normalization")

      image, bg_result = self.bg_normalizer.apply(image)

      results["techniques"]["background_normalization"] = bg_result

      print(f"   → Needed     : {bg_result['needed']}")
      print(f"   → Deviation  : {bg_result['deviation']:.2f}")

      if bg_result["needed"]:
          print("   Color Cast Removed\n")
      else:
          print("   ⏭ Skipped (No Cast)\n")


      # ==========================================================
      # Technique 10: Illumination Correction (CLAHE)
      # ==========================================================
      print("Technique 10: Illumination Correction (CLAHE)")

      detect_illum = self.illumination.detect(image)

      results["techniques"]["illumination"] = detect_illum

      print(f"   → Contrast STD    : {detect_illum['contrast_std']:.2f}")
      print(f"   → Threshold       : {detect_illum['threshold']}")
      print(f"   → Enhancement Needed : {detect_illum['needed']}")

      if detect_illum["needed"]:
          image = self.illumination.apply(image)

          output_path = save_output(
              image,
              self.output_dir,
              "illumination_clahe",
              "applied"
          )

          results["techniques"]["illumination"]["output_path"] = output_path

          print(f"   → Output Saved    : {output_path}")
          print("   CLAHE Applied\n")

      else:
          results["techniques"]["illumination"]["output_path"] = None
          print("  Illumination OK — No Enhancement Needed\n")

      # ==========================================================
      #  Technique 11: Noise Detection & Reduction
      # ==========================================================
      print(" Technique 11: Noise Detection & Reduction")

      detect_noise = self.noise.detect(image)

      results["techniques"]["noise_reduction"] = detect_noise

      print(f"   → Estimated Sigma : {detect_noise['estimated_sigma']:.4f}")
      print(f"   → Threshold       : {detect_noise['threshold']}")
      print(f"   → Noise Present   : {detect_noise['needed']}")

      if detect_noise["needed"]:
          image = self.noise.apply(image)

          output_path = save_output(
              image,
              self.output_dir,
              "noise_reduction",
              "applied"
          )

          results["techniques"]["noise_reduction"]["output_path"] = output_path

          print(f"   → Output Saved    : {output_path}")
          print(" Noise Reduction Applied\n")

      else:
          results["techniques"]["noise_reduction"]["output_path"] = None
          print(" Noise Level Acceptable — No Denoising Needed\n")

      # ==========================================================
      # Technique 12: Stroke Width Consistency Check
      # ==========================================================
      print(" Technique 12: Stroke Width Consistency Check")

      stroke_result = self.stroke_check.evaluate(image)

      results["techniques"]["stroke_width_check"] = stroke_result

      print(f"   → Mean Width       : {stroke_result['mean_width']:.2f}")
      print(f"   → Std Width        : {stroke_result['std_width']:.2f}")
      print(f"   → Deviation Ratio  : {stroke_result['deviation_ratio']:.3f}")
      print(f"   → Threshold        : {stroke_result['threshold']}")
      print(f"   → Valid            : {stroke_result['valid']}")

      if not stroke_result["valid"]:
          print("OCR Quality Risk: Stroke Inconsistency\n")
      else:
          print("Stroke Width Stable\n")


      print("=============== PIPELINE COMPLETED SUCCESSFULLY ===============\n")

      return image, results
    
   
    
    
    
    
    
    def run_1(self, image):

      results = {"techniques": {}}

      blur_result = self.blur.detect(image)
      results["techniques"]["blur_detection"] = blur_result

      if blur_result["needed"]:
          return image, results

      # Orientation Correction and skew correction
      image, orientation_results = correct_document_orientation(image)

      results["techniques"]["orientation"] = \
          orientation_results["orientation_correction"]

      # Boundary Detection
      detect_boundary = self.boundary.detect(image)
      apply_boundary = self.boundary.apply(image, detect_boundary)

      results["techniques"]["document_boundary"] = {
          **detect_boundary,
          "output_path": apply_boundary["output_path"]
      }

      # Early exit if no document
      if not detect_boundary["needed"]:
          results["techniques"]["perspective"] = {
              "needed": False,
              "reason": "No document detected",
              "output_path": None
          }
          return image, results

      corners = detect_boundary["corners"]

      # Perspective Detection
      detect_persp = self.perspective.detect(image, detect_boundary)

      if detect_persp["needed"]:
          apply_persp = self.perspective.apply(image, detect_boundary)
          image = apply_persp["processed_image"]
          output_path = apply_persp["output_path"]

      else:
          x, y, w, h = cv2.boundingRect(corners.astype("int32"))
          cropped = image[y:y+h, x:x+w]

          output_path = save_output(
              cropped,
              self.output_dir,
              "document_crop",
              "applied"
          )

          image = cropped

      results["techniques"]["perspective"] = {
          **detect_persp,
          "output_path": output_path
      }

      return image, results
    def run_(self, image):
      results = {"techniques": {}}

      # STEP 1: Boundary Detection
      detect_boundary = self.boundary.detect(image)
      apply_boundary = self.boundary.apply(image, detect_boundary)

      results["techniques"]["document_boundary"] = {
          **detect_boundary,
          "output_path": apply_boundary["output_path"]
      }

      # If no document found → stop early
      if not detect_boundary["needed"]:
          return image, results

      corners = detect_boundary["corners"]

      # STEP 2: Perspective Detection
      detect_persp = self.perspective.detect(image, detect_boundary)

      if detect_persp["needed"]:
          apply_persp = self.perspective.apply(image, detect_boundary)
          image = apply_persp["processed_image"]
          output_path = apply_persp["output_path"]

      else:
          # Crop bounding rectangle instead
          x, y, w, h = cv2.boundingRect(corners.astype("int32"))
          cropped = image[y:y+h, x:x+w]

          output_path = save_output(
              cropped,
              self.output_dir,
              "document_crop",
              "applied"
          )

          image = cropped

      results["techniques"]["perspective"] = {
          **detect_persp,
          "output_path": output_path
      }

      return image, results