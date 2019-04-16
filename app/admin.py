from django.contrib import admin

# Register your models here.
from django.contrib import admin
from django_summernote.admin import SummernoteModelAdmin
from .models import Stock

# Apply summernote to all TextField in model.
class SomeModelAdmin(SummernoteModelAdmin):  # instead of ModelAdmin
    summernote_fields = '__all__'

# Register your models here.
admin.site.register(Stock)