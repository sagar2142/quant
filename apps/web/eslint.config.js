// Frontend lint — the counterpart to ruff/mypy on the Python side.
//
// `package.json` had carried a `lint` script since the console was written, but
// no config and no eslint dependency, so `npm run lint` failed with "couldn't
// find an eslint.config.*" rather than linting anything. A check that claims to
// run and does not is worse than no check, which is the argument this codebase
// makes about alert rules and data feeds; it applies to its own tooling too.
//
// `tsc --noEmit` already runs in `build` and catches type errors. What it
// cannot see is the class of bug this adds: a `useEffect` whose dependency
// array disagrees with its body, which type-checks perfectly and then serves a
// stale price on a trading screen.

import js from "@eslint/js";
import reactHooks from "eslint-plugin-react-hooks";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist/**", "src-tauri/**", "node_modules/**"] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ["src/**/*.{ts,tsx}"],
    plugins: { "react-hooks": reactHooks },
    rules: {
      ...reactHooks.configs.recommended.rules,

      // An exhaustive-deps warning is the frontend's version of a stale read.
      // It is an error here rather than a warning because `--max-warnings 0`
      // would fail on it anyway, and an error says why.
      "react-hooks/exhaustive-deps": "error",

      // An unused variable after a refactor is usually a line that was meant
      // to be wired up. Leading underscore is the deliberate opt-out.
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
    },
  },
);
