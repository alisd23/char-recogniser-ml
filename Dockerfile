FROM python:3

# Install tensorflow CPU version for python 3
RUN pip3 install tensorflow
RUN python --version

# Install dependencies
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=9000

ENTRYPOINT ["python", "server.py"]
