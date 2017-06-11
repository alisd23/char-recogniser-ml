import cherrypy
from predict import predict
from database import connect
import os

port = os.environ.get('PORT')
DEFAULT_PORT = 9001

cherrypy.config.update({ 'server.socket_port': int(port) if port else DEFAULT_PORT })

# Connect to database
connect()

class Root():
  @cherrypy.expose
  @cherrypy.tools.json_in()
  @cherrypy.tools.json_out()
  def predict(self):
    data = cherrypy.request.json
    image = data['image']
    predictions, activations = predict(image)

    print(len(activations))
    print(len(activations[0]))
    print(len(activations[0][0]))
    return {
      'predictions': predictions,
      # 'activations': activations
    }

if __name__ == '__main__':
  cherrypy.quickstart(Root(), '/')
