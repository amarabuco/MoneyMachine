
from django.contrib import admin
from django.urls import include, path

# summernote settings
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('pages/', include('pages.urls')),
    path('admin/', admin.site.urls),
    path('', include('pages.urls')),
     path('summernote/', include('django_summernote.urls')),
]

if settings.DEBUG:
     urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
