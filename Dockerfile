FROM python:3

ADD attendencesystem.py /


RUN pip install numpy==1.16.2
RUN pip install opencv-python==4.0.0.21
RUN pip install python-tk

CMD [ "python", "./attendencesystem.py" ]
