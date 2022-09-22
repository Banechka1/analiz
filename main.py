import matplotlib.pyplot as plt
# import statistics

#Задание №1
f = open('viborka.txt')
vib = f.read().split(' ')
f.close()

viborka = []
try:
	for i in vib:
		viborka.append(int(i))
	if len(viborka) != 20:
		print('Объём выборки не равен 20.')
	else:
		print(f'Выборка:  {viborka}')

		#Варианты
		variants = []
		z = 0
		while z != max(viborka):
			z += 1
			variants.append(z)
		print(f'Варианты: {variants}')

		#Частоты N
		frequencies = []
		z = 0
		while z != max(viborka):
			z += 1
			frequencies.append(viborka.count(z))
		print(f'Частоты:  {frequencies}')
		

		#Выборочное среднее или Математематическое ожидание
		sample_average = sum(viborka) / len(viborka)
		print(f'Выборочное среднее: {sample_average}')

		#Мода
		max_frequencies = max(frequencies)
		index = frequencies.index(max_frequencies)
		moda = variants[index]
		print(f'Мода: {moda}')

		#Медиана и упорядоченная выборка
		sort_viborka = viborka
		sort_viborka.sort()
		print(f'Упорядоченная выборка: {sort_viborka}')
		median = (sort_viborka[9] + sort_viborka[10]) / 2
		print(f'Медиана: {median}')

		#Выбороная дисперсия
		sample_variance = 0
		h = 0
		while h != len(frequencies):
			sample_variance += frequencies[h] * (variants[h] - sample_average)**2
			h += 1
		sample_variance = round(sample_variance / (len(viborka) - 1), 4)
		# sample_variance = statistics.variance(viborka)
		print(f'Выборочная дисперсия: {sample_variance}')


		#Стандартное отклонение
		standard_deviation = round(sample_variance**0.5, 4)
		print(f'Стандартное отклонение: {standard_deviation}')


		#Коэффициент вариации
		coefficient_of_variation = round(standard_deviation / sample_average * 100, 4)
		print(f'Коэффициент вариации: {coefficient_of_variation}')


		#Размах
		scope = max(viborka) - min(viborka)
		print(f'Размах: {scope}')




#Задание №2
	e = round(sample_average - standard_deviation, 2)
	f = 3.5
	g = round(sample_average + standard_deviation, 2)

	if e < f < g:
		print(f'\n\n\n{e} < {f} < {g}\nОтвет: Да')
	else:
		print(f'\n\n\n{e} < {f} < {g}\nОтвет: Нет')

	#Графики
	plt.subplot(1, 2, 1)
	plt.title('Полигон')
	plt.plot(variants, frequencies, color='red')
	plt.subplot(1, 2, 2)
	plt.title('Гистограмма')
	plt.bar(variants, frequencies, color='red')
	plt.show()
		

except Exception:
	print('Вы ввели неверные значения!')
