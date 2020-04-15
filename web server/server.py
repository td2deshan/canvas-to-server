import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
#from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from binascii import a2b_base64


defaults.device = torch.device('cpu')


model_name = 'sec_train.pkl'

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
#app.add_middleware(HTTPSRedirectMiddleware)
app.mount('/static', StaticFiles(directory='static'))

#load model
async def setup_learner():
    try:
        learn = load_learner('.\models', model_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

#predictor start
loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())
	

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    #read canvas data
    img_data = await request.form()
    img_bytes = img_data['file']
    #decode
    img_bytes = img_bytes.replace("data:image/png;base64,", "")
    img_bytes = a2b_base64(img_bytes)
    img = open_image(BytesIO(img_bytes))
    prediction, tensor, prob = learn.predict(img)
    #print('prob', prob)

    return JSONResponse({'result': str(prediction)})


#start server
if __name__ == '__main__':
    uvicorn.run(app=app, host='127.0.0.1', port=5000, log_level="info")
    
