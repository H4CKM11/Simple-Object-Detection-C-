#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace cuda;

const float inputWidth = 640.0;
const float inputHeight = 640.0;
const float scoreThreshold = .5;
const float nmsThreshold = .45;
const float confidenceThreshold = .45;

const float fontScale = .7;
const int fontFace = FONT_HERSHEY_SIMPLEX;
const int thickness = 1;

Scalar black = Scalar(0, 0, 0);
Scalar blue = Scalar(255, 178, 50);
Scalar yellow = Scalar(0, 255, 255);
Scalar red = Scalar(0, 0, 255);


void drawLable(Mat& inputImage, string label, int left, int top)
{
	//Display the label at the top of the bounding box
	int baseLine = 0;
	Size labelSize = getTextSize(label, fontFace, fontScale, thickness, &baseLine);
	top = max(top, labelSize.height);

	//Gathering the cordinates for top left and bottom right
	Point topLeftCorner = Point(left, top);
	Point bottomRightCorner = Point(left + labelSize.width, top + labelSize.height + baseLine);

	//Creating the bounding box and adding text to it
	rectangle(inputImage, topLeftCorner, bottomRightCorner, black, FILLED);
	putText(inputImage, label, Point(left, top + labelSize.height), fontFace, fontScale, yellow, thickness);

}

vector<Mat> preProcess(Mat& inputImage, Net& net)
{
	//Creating a Blob from the input image
	Mat blob;
	blobFromImage(inputImage, blob, 1. / 255., Size(inputWidth, inputHeight), Scalar(), true, false);

	//Setting the input of the net with the Blob object we just created
	net.setInput(blob);

	//Running the forward pass to get output of the output layers
	vector<Mat> outputs;
	net.forward(outputs, net.getUnconnectedOutLayersNames());

	return outputs;
}

Mat postProcess(Mat& inputImage, vector<Mat> outputs, const vector<string>& className)
{
	//Initializing the respected vectors if a good detection is found
	vector<int> classIDs;
	vector<float> confidences;
	vector<Rect> boxes;
	int count = 0;

	//Since YOLOv5 takes in 640x640 we needed to scale down the image in the pre-process stage, here we are resizing them back
	float xFactor = inputImage.cols / inputWidth;
	float yFactor = inputImage.rows / inputHeight;
	float* data = (float*) outputs [0].data;
	const int dimensions = 85;

	//Yolov5 architech, head, has 3 output layers, each layer detects different objects at different size. Small, medium, and large.
	//The dimensions of each layer goes as follow. 80x80, 40x40, 20x20. With Yolo's input being 640, the total grid cells required
	//is 25200. (3*80*80) + (3*40*40) + (3*20*20) = 25200
	const int rows = 25200;

	//We are going to interate through every row one by one, saving the good detections and discarding the bad
	for (int i = 0; i < rows; ++i)
	{
		float confidence = data [4];

		if (confidence >= confidenceThreshold)
		{

			float* classScores = data + 5;

			//We are creating a new Mat object 1x85 and storing class score of 80 classes. The trained model has 80 objects it can detect.
			Mat scores(1, className.size(), CV_32FC1, classScores);

			//Perform a minMaxLocation. minMaxLoc scans through a single channel array and locates the global min and max. in our case scores.
			Point classID;
			double maxClassScore = 0;
			minMaxLoc(scores, 0, &maxClassScore, 0, &classID);

			//If a detection is found, we see if the detection score is greater than the threshold we placed 
			if (maxClassScore > scoreThreshold)
			{
				count++;
				//If confidence is greater than threshold we put our results in the vectors we created above
				confidences.push_back(confidence);
				classIDs.push_back(classID.x);

				//We find the center of the detected object
				float centerX = data [0];
				float centerY = data [1];

				//We find the box dimensions of the detected object
				float fWidth = data [2];
				float fHeight = data [3];

				//We create the bounding box cordinates
				int left = int(( centerX - 0.5 * fWidth ) * xFactor);
				int top = int(( centerY - 0.5 * fHeight ) * yFactor);
				int width = int(fWidth * xFactor);
				int height = int(fHeight * yFactor);

				//store the bounding box in the vector
				boxes.push_back(Rect(left, top, width, height));
			}
		}

		data += 85;
	}

	//After finding a good detection, we are left with the bounding boxes. However we can sometimes be left with mutliple bounding boxes. To remove
	//these, we can call the NMSBoxes() function. The NMSBoxes function calculates the Intersection Over Union on of all the bounding boxes,
	//compares it to the nmsTheshold above and spits out any boxes that are greater than the theshold
	vector<int> indices;
	NMSBoxes(boxes, confidences, scoreThreshold, nmsThreshold, indices);

	for (int i = 0; i < indices.size(); i++)
	{
		int index = indices [i];
		Rect box = boxes [index];
		int left = box.x;
		int top = box.y;
		int boxWidth = box.width;
		int boxHeight = box.height;

		//Draw the bounding box
		rectangle(inputImage, Point(left, top), Point(left + boxWidth, top + boxHeight), blue, 3 * thickness);

		//Create the label for the class name and confidence
		string label = format("%.2f", confidences [index]);
		label = className [classIDs [index]] + ":" + label;

		//Draw the lable on the final product
		drawLable(inputImage, label, left, top);
	}

	cout << count;

	return inputImage;
}

int main()
{
	//Load the class names into a vector
	vector<string> classNames;
	ifstream openFile("classes.txt");

	if (openFile.is_open())
	{
		cout << "File is open";
	}
	string line;

	while (getline(openFile, line))
	{
		classNames.push_back(line);
		cout << classNames [0];
	}

	//Load Net
	Net net;
	net = readNet("YOLOv5s.onnx");

	net.setPreferableBackend(DNN_BACKEND_CUDA);
	net.setPreferableTarget(DNN_TARGET_CUDA);


	//Load in image
	VideoCapture videoCap("sample.mp4");

	while (videoCap.isOpened())
	{
		Mat frame;
		bool isSuccess = videoCap.read(frame);

		vector<Mat> detections;
		detections = preProcess(frame, net);
		Mat frameCloned = frame.clone();
		Mat image = postProcess(frameCloned, detections, classNames);

		vector<double> layersTimes;
		double freq = getTickFrequency() / 1000;
		double t = net.getPerfProfile(layersTimes) / freq;
		string label = format("Inference time : %.2f ms", t);
		putText(image, label, Point(20, 40), fontFace, fontScale, red);
		imshow("Output", image);
		waitKey(10);
	}

	return 0;
}