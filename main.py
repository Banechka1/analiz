import pandas as pd
import numpy as np

#Загружаю датасеты
ptd1 = pd.read_csv('part_01.csv')
ptd2 = pd.read_csv('part_02.csv')
ptd3 = pd.read_csv('part_03.csv')
ptd4 = pd.read_csv('part_04.csv')
ptd5 = pd.read_csv('part_05.csv')
ptd6 = pd.read_csv('part_06.csv')
ptd7 = pd.read_csv('part_07.csv')
ptd8 = pd.read_csv('part_08.csv')
ptd9 = pd.read_csv('part_09.csv')
ptd10 = pd.read_csv('part_10.csv')
ptd11 = pd.read_csv('part_11.csv')
ptd12 = pd.read_csv('part_12.csv')

#Объединяю в один
btm1 = pd.concat([ptd5, ptd12], ignore_index=True)
btm2 = pd.concat([ptd6, ptd10], ignore_index=True)
btm3 = pd.concat([ptd2, ptd11], ignore_index=True)
btm4 = pd.concat([ptd1, ptd9], ignore_index=True)
btm5 = pd.concat([ptd3, ptd8], ignore_index=True)
btm6 = pd.concat([ptd4, ptd7], ignore_index=True)

final_dataset = btm1.merge(btm2, left_on='ID', right_on='ID')
final_dataset = final_dataset.merge(btm3, left_on='ID', right_on='ID')
final_dataset = final_dataset.merge(btm4, left_on='ID', right_on='ID')
final_dataset = final_dataset.merge(btm5, left_on='ID', right_on='ID')
final_dataset = final_dataset.merge(btm6, left_on='ID', right_on='ID')
final_dataset = final_dataset.sort_values(by='ID')

#Дропаю дубликаты
final_dataset = final_dataset.drop_duplicates(keep=False)

#Первые 10 строк объединенного датасета
print(final_dataset.head(10))
#Количество столбцов и колонок в объединенном датасете
print(final_dataset.shape)

#Найти в получившемся датасете все поля, которые являются текстовыми
print(final_dataset.info(verbose=True))
#Список из названия колонок, где все поля являются текстовыми
columns_dtype_object = list(final_dataset.select_dtypes(['object']).columns)
#Привести их значения к нижнему регистру
for column in columns_dtype_object:
	final_dataset[column] = final_dataset[column].str.lower()

#Если текстовое значение состоит только из “ “ (пробел), то его заменить на NaN
for column in columns_dtype_object:
	final_dataset[column].replace(' ', np.NaN)

#Если в текстовом значении есть цифры (телефон, дата рождения и др.), то всё значение текстового поля тоже заменить на NaN.
for column in columns_dtype_object:
	if column == 'Pack':
		pass
	else:
		for i in final_dataset[column]:
			try:
				final_dataset[column].where((i.isdigit() == True), other=np.NaN, inplace=True)
			except Exception as e:
				pass


#Выведите количество уникальных значения в полях CLNT_TRUST_RELATION, APP_MARITAL_STATUS, CLNT_JOB_POSITION
print(len(final_dataset['CLNT_TRUST_RELATION'].unique()))
print(len(final_dataset['APP_MARITAL_STATUS'].unique()))
print(len(final_dataset['CLNT_JOB_POSITION'].unique()))

#Выведите колонки итогового датасета
print(final_dataset.columns)