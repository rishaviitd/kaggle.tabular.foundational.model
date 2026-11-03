"""Optional ASGI transport sharing the existing single GPU worker."""
import asyncio


def create_async_app(service):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.post('/predict')
    async def predict(request: Request):
        try:
            size = int(request.headers.get('content-length', 0))
            if not 0 < size <= 4_000_000:
                raise ValueError('Invalid request size')
            body = await request.json()
            predictions = await asyncio.wrap_future(service.submit(body['records']))
            return JSONResponse({'predictions': predictions})
        except (ValueError, KeyError, TypeError) as error:
            return JSONResponse({'error': str(error)}, status_code=400)

    return app
