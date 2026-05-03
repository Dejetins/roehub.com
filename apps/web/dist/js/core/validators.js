import { translate } from "./locale.js";

export function required(value, message = translate("js.validation.required")) {
  const valid = value !== null && value !== undefined && String(value).trim() !== "";
  return valid ? null : message;
}

export function number(value, message = translate("js.validation.number")) {
  if (required(value) !== null) {
    return message;
  }
  return Number.isFinite(Number(value)) ? null : message;
}

export function validateFields(fields, rules) {
  const errors = {};
  Object.entries(rules).forEach(([fieldName, validators]) => {
    for (const validator of validators) {
      const error = validator(fields[fieldName]);
      if (error) {
        errors[fieldName] = error;
        break;
      }
    }
  });
  return errors;
}

export function hasErrors(errors) {
  return Object.keys(errors).length > 0;
}
