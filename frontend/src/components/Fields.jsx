import React from "react";

export function Field({ label, hint, children }) {
  return (
    <label className="field">
      <span className="field-label">
        {label}
        {hint && <span className="field-hint" title={hint}> ⓘ</span>}
      </span>
      {children}
    </label>
  );
}

export function NumberField({ label, value, onChange, min, max, step = 1, hint }) {
  return (
    <Field label={label} hint={hint}>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(e.target.value === "" ? "" : Number(e.target.value))}
      />
    </Field>
  );
}

export function SelectField({ label, value, onChange, options, hint }) {
  return (
    <Field label={label} hint={hint}>
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => (
          <option key={o.value} value={o.value} disabled={o.disabled}>
            {o.label}
          </option>
        ))}
      </select>
    </Field>
  );
}

export function Toggle({ label, checked, onChange, hint }) {
  return (
    <label className="toggle">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span>{label}</span>
      {hint && <span className="field-hint" title={hint}> ⓘ</span>}
    </label>
  );
}

export const RESOLUTIONS_VIDEO = [
  { value: "832x480", label: "832 × 480 (480p quer)" },
  { value: "480x832", label: "480 × 832 (480p hoch)" },
  { value: "1280x720", label: "1280 × 720 (720p quer)" },
  { value: "720x1280", label: "720 × 1280 (720p hoch)" },
];

export const RESOLUTIONS_IMAGE = [
  { value: "1024x1024", label: "1024 × 1024 (1:1)" },
  { value: "1344x768", label: "1344 × 768 (16:9)" },
  { value: "768x1344", label: "768 × 1344 (9:16)" },
  { value: "1216x832", label: "1216 × 832 (3:2)" },
  { value: "832x1216", label: "832 × 1216 (2:3)" },
];
