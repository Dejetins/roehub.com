export function formatPercent(value, options = {}) {
  const locale = options.locale || "en";
  const digits = options.digits ?? 2;
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return options.empty || "0.00%";
  }
  const sign = numeric > 0 ? "+" : "";
  return `${sign}${numeric.toLocaleString(locale, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })}%`;
}

export function formatNumber(value, options = {}) {
  const locale = options.locale || "en";
  const digits = options.digits ?? 2;
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return options.empty || "-";
  }
  return numeric.toLocaleString(locale, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function formatDateTime(value, options = {}) {
  const locale = options.locale || "en";
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) {
    return options.empty || "-";
  }
  return new Intl.DateTimeFormat(locale, {
    dateStyle: "medium",
    timeStyle: "short",
    ...options.format,
  }).format(date);
}

export function financialClassName(value) {
  const numeric = Number(value);
  if (numeric > 0) {
    return "rh-financial--positive";
  }
  if (numeric < 0) {
    return "rh-financial--negative";
  }
  return "rh-financial--neutral";
}
