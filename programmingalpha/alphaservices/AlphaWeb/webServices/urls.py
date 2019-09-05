from django.urls import path
from . import views

urlpatterns = [
    path('alpha-QA', views.getAnswer, name='ask question'),
    path('index', views.index, name='ask question')

]
