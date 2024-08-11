import cv2
import boto3

image_path = "./road.jpeg"

def get_image(imgpath):
    with open(imgpath, "rb") as img:
        return img.read()

def object_detection(imgpath):
    image_contain = get_image(imgpath)
    image = cv2.imread(imgpath)
    width, height = image.shape[1], image.shape[0]
    print("Processing Start....")
    client = boto3.client("rekognition")
    response = client.detect_labels(Image={"Bytes": image_contain}, MaxLabels=10)
    print("Building the block")
    for label in response["Labels"]:
        if label["Confidence"] > 80:
            label_name = label["Name"]
            instances = label["Instances"]
            for instance in instances:
                bounding_box = instance["BoundingBox"]
                x = int(bounding_box["Left"] * width)
                y = int(bounding_box["Top"] * height)
                w = int(bounding_box["Width"] * width)
                h = int(bounding_box["Height"] * height)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, label_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)
    return image

result = object_detection(image_path)
while True:
    cv2.imshow("Result", result)
    if cv2.waitKey(1) == ord("q"):
        break