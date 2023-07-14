import streamlit as st
import boto3
import os
import tempfile


def upload_document_to_s3(bucket_name, folder_name, file_path):
    s3 = boto3.client('s3')
    key = folder_name + '/' + file_path
    s3.upload_file(file_path, bucket_name, key)
    return key


def detect_text(photo, bucket):
    temp = ""
    session = boto3.Session(profile_name='default')
    client = session.client('rekognition')

    response = client.detect_text(Image={'S3Object': {'Bucket': bucket, 'Name': photo}})

    textDetections = response['TextDetections']
    for text in textDetections:
        temp = temp + " " + text['DetectedText']
    return temp


def detect_medical_entities(text):
    client = boto3.client(service_name='comprehendmedical', region_name='ap-southeast-2')
    result = client.detect_entities(Text=text)
    entities = result['Entities']
    entity_groups = {}
    for entity in entities:
        entity_type = entity['Type']
        entity_text = entity['Text']

        if entity_type not in entity_groups:
            entity_groups[entity_type] = set()
        entity_groups[entity_type].add(entity_text)

    return entity_groups


def main():
    st.title("Document Text Analysis")
    file = st.file_uploader("Upload a Document")
    print(file)

    if file is not None:
        bucket = 'careallianzmobileapps3bucket181010-dev'
        folder_name = 'public'
        temp_dir = tempfile.TemporaryDirectory()
        temp_file_path = os.path.join(temp_dir.name, file.name)
        with open(temp_file_path, 'wb') as f:
            f.write(file.getbuffer())

        photo = upload_document_to_s3(bucket, folder_name, temp_file_path)
        text = detect_text(photo, bucket)
        print(text)
        st.subheader("Detected Text:")
        st.text(text)
        
        entities = detect_medical_entities(text)
        print(entities)
        st.subheader("Medical Entities:")
        for entity_type, entity_texts in entities.items():
            st.markdown(f"{entity_type}:")
            for entity_text in entity_texts:
                st.text(entity_text)
            st.markdown("---")


if __name__ == "__main__":
    main()
