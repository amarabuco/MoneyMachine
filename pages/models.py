from django.db import models

# Create your models here.
class QuestionType(models.Model):
    type_text = models.CharField(max_length=250)

class Question(models.Model):
    question_type = models.ForeignKey(QuestionType, on_delete=models.CASCADE)
    question_text = models.CharField(max_length=500)

class Answer(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    answer_text = models.TextField()


