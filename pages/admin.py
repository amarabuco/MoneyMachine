from django.contrib import admin
from django_summernote.admin import SummernoteModelAdmin
from .models import Question, QuestionType, Answer

# Apply summernote to all TextField in model.
class SomeModelAdmin(SummernoteModelAdmin):  # instead of ModelAdmin
    summernote_fields = '__all__'

# Register your models here.
admin.site.register(Question, SomeModelAdmin)
admin.site.register(QuestionType, SomeModelAdmin)
admin.site.register(Answer, SomeModelAdmin)