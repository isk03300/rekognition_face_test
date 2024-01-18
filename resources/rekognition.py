from flask import request
from flask_restful import Resource
from config import Config
from mysql.connector import Error
from datetime import datetime
import boto3
import json
import os
from PIL import Image, ImageDraw
import io
from io import BytesIO

# 얼굴 인식 (사진 1장)
class rekognitionFaceResource(Resource) :
    
    def post(self) :

        file = request.files.get("photo")

        if file is None :
            return {"error" : "파일이 존재하지 않습니다."}, 400
        
        current_time = datetime.now()

        new_file_name = current_time.isoformat().replace(":", "_") + ".jpg"

        file.filename = new_file_name

        s3 = boto3.client("s3", aws_access_key_id = Config.AWS_ACCESS_KEY_ID,
                     aws_secret_access_key = Config.AWS_SECRET_ACCESS_KEY)

        try :
            s3.upload_fileobj(file, Config.S3_BUCKET, file.filename, 
                              ExtraArgs = {"ACL" : "public-read",
                                           "ContentType" : "image/jpeg"})

        except Exception as e :
            print(e)
            return {"error" : str(e)}, 500
        
        # S3에 이미지가 있으니 rekognition 이용해서 object detection 한다.

        faces_len = self.detect_faces(new_file_name, Config.S3_BUCKET)
        
        return {"result" : "success", "FaceDetails" : faces_len}, 200

    def detect_faces(self, photo, bucket) :

        client = boto3.client('rekognition', region_name='ap-northeast-2', 
                              aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
                              aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY)

        response = client.detect_faces(Image={'S3Object':{'Bucket':bucket,'Name':photo}}, Attributes=['ALL'])

        s3_connection = boto3.resource('s3')
        s3_object = s3_connection.Object(Config.S3_BUCKET, photo)
        s3_response = s3_object.get()

        stream = io.BytesIO(s3_response['Body'].read())
        image = Image.open(stream)

        imgWidth, imgHeight = image.size
        draw = ImageDraw.Draw(image)

        for faceDetail in response['FaceDetails']:
            print(str(faceDetail['AgeRange']['Low'])+ '세에서 ' + str(faceDetail['AgeRange']['High']) + '세 사이의 얼굴이 감지되었습니다.')

            # print('다음 속성들 : ')
            # print(json.dumps(faceDetail, indent=4, sort_keys=True))

            # 개별 얼굴 세부 정보에 대한 예측에 엑세스하고 출력
            print("성별 : " + str(faceDetail['Gender']))
            print("미소 : " + str(faceDetail['Smile']))
            print("안경 : " + str(faceDetail['Eyeglasses']))
            print("얼굴 가려짐 : " + str(faceDetail['FaceOccluded']))
            print("감정 : " + str(faceDetail['Emotions'][0]))

            box = faceDetail['BoundingBox']
            left = imgWidth * box['Left']
            top = imgHeight * box['Top']
            width = imgWidth * box['Width']
            height = imgHeight * box['Height']

            points = (
            (left, top),
            (left + width, top),
            (left + width, top + height),
            (left, top + height),
            (left, top)
            )

            draw.line(points, fill='#FF0000', width=2)

            # 소수 변환
            faceDetail['Emotions'][0]['Confidence'] = round(float(faceDetail['Emotions'][0]['Confidence']), 3)
            faceDetail['Gender']['Confidence'] = round(float(faceDetail['Gender']['Confidence']), 3)

            info_text = f"AgeRange : {str(faceDetail['AgeRange']['Low'])} ~ {str(faceDetail['AgeRange']['High'])}\n Gender : {faceDetail['Gender']['Value']} {faceDetail['Gender']['Confidence']}% \n Emotion : {faceDetail['Emotions'][0]['Type']} {faceDetail['Emotions'][0]['Confidence']}%"
            draw.text((left + width + 5, top), info_text, fill='#FF0000')

        image.show()

        # print("얼굴 세부 정보 : " + str(response['FaceDetails']));    

        return len(response['FaceDetails'])
    
# 얼굴 비교 (사진 2장)    
class rekognitionFaceCompareResource(Resource) :

    def post(self) : 
        sourceFile = request.files.get("sourceFile")
        targetFile = request.files.get("targetFile")

        if sourceFile is None or targetFile is None :
            return {"error" : "파일 두개를 올려주세요."}, 400

        FaceMatches = self.compare_faces(sourceFile, targetFile)
        
        return {"result" : "success", "FaceMatches" : FaceMatches}, 200
    
    def compare_faces(self, sourceFile, targetFile) :

        client = boto3.client('rekognition', region_name='ap-northeast-2', 
                                aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
                                aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY)
        
        imageSource = BytesIO(sourceFile.read())
        imageTarget = BytesIO(targetFile.read())

        response = client.compare_faces(SimilarityThreshold=0, SourceImage={'Bytes': imageSource.read()}, TargetImage={'Bytes': imageTarget.read()})

        # 이미지를 열어 얼굴 좌표를 얻어올 수 있도록 처리
        imageTarget.seek(0)
        target_image = Image.open(imageTarget)
        draw = ImageDraw.Draw(target_image)

        similarity_list = []
        for faceMatch in response['FaceMatches']:
            position = faceMatch['Face']['BoundingBox']
            similarity = str(faceMatch['Similarity'])

            similarity_list.append(similarity)

            # 얼굴 좌표를 픽셀 단위로 변환
            imgWidth, imgHeight = target_image.size
            left = imgWidth * position['Left']
            top = imgHeight * position['Top']
            width = imgWidth * position['Width']
            height = imgHeight * position['Height']

            points = (
            (left, top),
            (left + width, top),
            (left + width, top + height),
            (left, top + height),
            (left, top)
            )

            draw.line(points, fill='#ff0000', width=2)

            # 일치 확률을 이미지에 표시
            similarity = round(float(similarity), 3)
            draw.text((left + width + 5, top), f"similarity : {similarity}%", fill='#ff0000')

            # print('발견된 얼굴 위치 : ' +
            #     str(position['Left']) + ' ' +
            #     str(position['Top']) +
            #     ' 일치 확률 : ' + similarity + '% confidence')
            
        target_image.show()

        imageSource.close()
        imageTarget.close()

        return similarity_list
        # return response['FaceMatches']

        
