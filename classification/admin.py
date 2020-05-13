from django.contrib import admin
from classification.models import DataSet,DataSetModel
# Register your models here.

class DataSetAdmin(admin.ModelAdmin):
    list_display = ['id', 'title', 'train','test','val','categories','vocab_filename','vector_word_filename','vector_word_npz','desc','is_hide','is_del',]
admin.site.register(DataSet, DataSetAdmin)
# Register your models here.
