import os
import argparse
from processor import Processor

from transformers.load_images import LoadImages
from transformers.detect_faces import DetectFaces
from transformers.save_faces import SaveFaces
from transformers.save_summary import SaveSummary
from transformers.display_summary import DisplaySummary


def parse_args():

    # Parse command line arguments
    ap = argparse.ArgumentParser(description="Image processing pipeline")
    ap.add_argument("-i", "--input", required=True,
                    help="path to input image files")
    ap.add_argument("-o", "--output", default="output",
                    help="path to output directory")
    ap.add_argument("-os", "--out-summary", default="summary.json",
                    help="output JSON summary file name")
    ap.add_argument("--prototxt", default="./models/face_detector/deploy.prototxt.txt",
                    help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("--model", default="./models/face_detector/res10_300x300_ssd_iter_140000.caffemodel",
                    help="path to Caffe pre-trained model")
    ap.add_argument("--confidence", type=float,  default=0.5,
                    help="minimum probability to filter weak face detections")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="face detection batch size")

    return ap.parse_args()
                  

def main(args):
    # Create pipeline steps

    load_images = LoadImages(args.input)

    detect_faces = DetectFaces(prototxt=args.prototxt, model=args.model,
                               confidence=args.confidence, batch_size=args.batch_size)

    save_faces = SaveFaces(args.output)

    summary_file = os.path.join(args.output, args.out_summary)
    save_summary = SaveSummary(summary_file)

    display_summary = DisplaySummary()

    # Create image processing pipeline
    pipeline = (load_images |
                detect_faces |
                save_faces |
                save_summary |
                display_summary)
    
    # Create processor for processing pipeline
    process=Processor(pipeline)
    try:
        process.run(verbose=True)
    except:
        return
    finally:
            print(f"[INFO] Saving summary to {summary_file}...")
            save_summary.write()

if __name__ == "__main__":
    #args = parse_args()    # Disable during debugging 
    args=argparse.Namespace(
        input="assets/images", 
        output="output",
        out_summary="output.json",
        prototxt="./models/face_detector/deploy.prototxt.txt",
        model="./models/face_detector/res10_300x300_ssd_iter_140000.caffemodel",
        confidence=0.5,
        batch_size=1)   # Disable when run through terminal

    main(args)