import lightly_train
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes


if __name__ == "__main__":
    # Use the DINOv3 provided by LightlyTrain
    model = lightly_train.load_model("dinov3/convnext-small-ltdetr-coco")

    prediction = model.predict("examples/image.jpg")

    image = read_image("examples/image.jpg")
    image_with_boxes = draw_bounding_boxes(
        image,
        boxes=prediction["bboxes"],
        labels=[model.classes[label.item()] for label in prediction["labels"]],
    )
    plt.imshow(image_with_boxes.permute(1, 2, 0))
    plt.show()
    print(prediction)
    print([model.classes[predi] for predi in prediction["labels"].tolist()])
        
