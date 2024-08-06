class PrintablePolynomial:
    def __init__(self, coeffs):
        self.coeffs = coeffs
        self.function = lambda x: self.polynomial(x, coeffs)

    @staticmethod
    def polynomial(x, coeffs):
        result = 0
        for i in range(len(coeffs)):
            result += coeffs[i] * x ** i
        return result

    def __str__(self):
        if not self.coeffs:
            return '0.0'
        result = []
        def append_coeff(idx, suffix):
            coeff = self.coeffs[idx]
            if coeff > 0:
                result.append(f'+ {coeff}{suffix} ')
            if coeff < 0:
                result.append(f'- {abs(coeff)}{suffix} ')

        for i in range(len(self.coeffs) - 1, 1, -1):
            append_coeff(i, f'x^{i}')
        if len(self.coeffs) >= 2:
            append_coeff(1, 'x')
        if len(self.coeffs) >= 1:
            append_coeff(0, '')
        if result[0][0] == '-':
            result[0] = '-' + result[0][2:]
        return ''.join(result).strip('+ ')
