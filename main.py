import model
import sys
from inference import infer


if __name__ == "__main__":
    if len(sys.argv) == 1:
        model.train()
    elif len(sys.argv) == 3:
        model_pickle = sys.argv[1]
        image = sys.argv[2]
        infer(model_name=model_pickle, image_name=image)
    else:
        print("incorrect number of args")
