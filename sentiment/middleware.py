from django.core.management import call_command

class FlushDatabaseMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        call_command('flush', interactive=False)
        response = self.get_response(request)
        return response
