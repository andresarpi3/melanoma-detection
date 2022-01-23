FROM public.ecr.aws/lambda/python:3.7


COPY requirements.txt  .
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

COPY app/app.py ${LAMBDA_TASK_ROOT}
COPY app/model.py ${LAMBDA_TASK_ROOT}
ADD ./weights/ ${LAMBDA_TASK_ROOT}/weights

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.handler" ] 
