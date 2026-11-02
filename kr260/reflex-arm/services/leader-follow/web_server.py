"""LAN page and same-origin API proxy over the dedicated private SSH tunnel."""
import argparse
from pathlib import Path
from aiohttp import ClientSession,ClientTimeout,web,ClientError


def create_app(root,host,upstream):
    expected='http://'+host
    @web.middleware
    async def boundary(request,handler):
        if request.host!=host:raise web.HTTPForbidden(text='Unexpected Host')
        if request.method=='POST':
            if request.headers.get('Origin')!=expected or request.content_type!='application/json':
                raise web.HTTPForbidden(text='Same-origin JSON required')
        response=await handler(request)
        response.headers.update({'X-Content-Type-Options':'nosniff','X-Frame-Options':'DENY',
            'Referrer-Policy':'same-origin','Cache-Control':'no-store',
            'Content-Security-Policy':"default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'"})
        return response
    app=web.Application(middlewares=[boundary],client_max_size=16384)
    async def lifecycle(app):
        app['upstream_session']=ClientSession(timeout=ClientTimeout(total=3,sock_connect=1))
        yield
        await app['upstream_session'].close()
    app.cleanup_ctx.append(lifecycle)
    async def proxy(request):
        allowed={'/api/status','/api/health','/api/export.csv','/api/reference','/api/mapping','/api/preview/start','/api/preview/stop','/api/follow/start','/api/follow/stop','/api/follow/heartbeat'}
        allowed.add('/api/leader/recheck')
        if request.path not in allowed:raise web.HTTPNotFound()
        try:
            async with app['upstream_session'].request(request.method,upstream+request.path,
                data=await request.read(),headers={'Content-Type':'application/json','X-Reflex-Proxy':'leader-web'}) as response:
                headers={k:response.headers[k] for k in ['Content-Type','Content-Disposition'] if k in response.headers}
                return web.Response(body=await response.read(),status=response.status,headers=headers)
        except (ClientError,TimeoutError):
            return web.json_response({'error':'KR260连接暂不可用；无法确认电机状态。跟随端会按心跳超时停止J1，请查看现场状态。'},status=503)
    async def static(request):
        name='index.html' if request.path=='/' else request.path.lstrip('/')
        if name not in ['index.html','app.js','style.css']:raise web.HTTPNotFound()
        return web.FileResponse(root/name)
    app.router.add_route('GET','/api/{path:.*}',proxy);app.router.add_route('POST','/api/{path:.*}',proxy)
    app.router.add_get('/{path:.*}',static)
    return app


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bind',default='172.21.16.96');p.add_argument('--port',type=int,default=3002);a=p.parse_args()
    web.run_app(create_app(Path(__file__).parent/'dist',f'{a.bind}:{a.port}','http://127.0.0.1:19010'),host=a.bind,port=a.port,access_log=None)
