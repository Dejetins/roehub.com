export function formatPercent(value, { maximumFractionDigits = 2 } = {}) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "0.00%";
  }
  return `${number.toFixed(maximumFractionDigits)}%`;
}

export function financialClass(value) {
  const number = Number(value);
  if (number > 0) {
    return "rh-financial--positive";
  }
  if (number < 0) {
    return "rh-financial--negative";
  }
  return "rh-financial--neutral";
}

export function formatDateTime(value, locale = document.documentElement.lang || "en") {
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) {
    return "";
  }
  return new Intl.DateTimeFormat(locale, {
    dateStyle: "short",
    timeStyle: "medium",
  }).format(date);
}
