package main

import (
	"context"
	"fmt"
	// import xxx as vision
	vision "cloud.google.com/go/vision/apiv1"
)

// Annotate an image file based on Cloud Vision API, return score and error if exists.
func annotate(uri string) (float32, error) {
	ctx := context.Background()
	client, err := vision.NewImageAnnotatorClient(ctx)
	if err != nil {
		return 0.0, err
	}
	defer client.Close()
	// input a uri to access image
	image := vision.NewImageFromURI(uri)
	// detect only one face so only one result in face anno array
	faceAnnotations, err := client.DetectFaces(ctx, image, nil, 1)
	if err != nil {
		return 0.0, err
	}
	if len(faceAnnotations) == 0 {
		fmt.Println("No faces found.")
		return 0.0, nil
	}
	return faceAnnotations[0].DetectionConfidence, nil
}
