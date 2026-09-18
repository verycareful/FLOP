# Documentation Guide

How everything under `docs/` is written. This page is the single source for
the tree layout, the sections each page carries and the formatting rules;
change it here rather than in any instruction file, so the rules a
contributor can read are the rules they are held to.

## Tree

```text
docs/
  MasterDocumentation.md   this page
  Architecture.md          the design: concepts, facade, how an algorithm is added
  api/                     one page per public header
  algorithms/              one page per algorithm, the method as implemented
```

## Page types

An API page documents one public header, in this order: purpose, include
line and namespace, types and functions in declaration order with their
contracts, the exceptions the header can throw, and one runnable example.

An algorithm page documents one method as FLOP implements it: the source
(paper, section), the method in the paper's terms, every deviation from the
paper and why, the options the method reads, the stopping rules it honours,
and its known limits.

## Formatting

- A plain `# Title` on the first line. Headings in order, no skipped levels.
- Fenced code blocks carry a language tag: `cpp`, `bash`, `cmake`, `text`.
- No release or version numbers in prose. A page describes the code as it
  stands; the CHANGELOG carries history.
- No em dashes and no double hyphens. Use a colon, a comma, parentheses or
  two sentences.
- Tables where the content is a grid; bullets for lists; numbered lists for
  ordered steps.
- Nothing aspirational: a page describes what exists. Planned work is not
  documented until it ships.
- A page never cites a path a reader cloning the repository would not find.
